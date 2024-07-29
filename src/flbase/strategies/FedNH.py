from collections import OrderedDict, Counter
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import torch
try:
    import wandb
except ModuleNotFoundError:
    pass
from ..server import Server
from ..client import Client
from ..models.CNN import *
from ..models.MLP import *
from ..utils import setup_optimizer, linear_combination_state_dict, setup_seed
from ...utils import autoassign, save_to_pkl, access_last_added_element
from .FedUH import FedUHClient, FedUHServer
import math
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import os
import plotly.express as px
from scipy.spatial import ConvexHull
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix
import seaborn as sns



class FedNHClient(FedUHClient):
    """Client class for Federated NH (FedNH) algorithm."""
    def __init__(self, criterion, trainset, testset, client_config, cid, device, **kwargs):
        """
        Initializes a FedNH client.

        Args:
        - criterion (torch.nn.Module): Loss function criterion for training.
        - trainset (torch.utils.data.Dataset): Training dataset for the client.
        - testset (torch.utils.data.Dataset): Test dataset for the client.
        - client_config (dict): Configuration dictionary for client parameters.
        - cid (int): Client ID.
        - device (torch.device): Device (CPU or GPU) where the client operates.
        - **kwargs: Additional keyword arguments.

        Attributes:
        - count_by_class_full (torch.Tensor): Full count of samples per class for prototype estimation.
        """
        super().__init__(criterion, trainset, testset,
                         client_config, cid, device, **kwargs)
        temp = [self.count_by_class[cls] if cls in self.count_by_class.keys() else 1e-12 for cls in range(client_config['num_classes'])]
        self.count_by_class_full = torch.tensor(temp).to(self.device)

    def _estimate_prototype(self):
        """
        Estimates prototype vectors based on class distributions.

        Returns:
        - to_share (dict): Dictionary containing scaled_prototype and count_by_class_full.
        - scaled_prototype (torch.Tensor): Scaled prototype vectors.
        - count_by_class_full (torch.Tensor): Full count of samples per class.
        """
        self.model.eval()
        self.model.return_embedding = True
        embedding_dim = self.model.prototype.shape[1]
        prototype = torch.zeros_like(self.model.prototype)
        with torch.no_grad():
            for i, (x, y) in enumerate(self.trainloader):
                # forward pass
                x, y = x.to(self.device), y.to(self.device)
                # feature_embedding is normalized
                feature_embedding, _ = self.model.forward(x)
                classes_shown_in_this_batch = torch.unique(y).cpu().numpy()
                for cls in classes_shown_in_this_batch:
                    mask = (y == cls)
                    feature_embedding_in_cls = torch.sum(feature_embedding[mask, :], dim=0)
                    prototype[cls] += feature_embedding_in_cls
        for cls in self.count_by_class.keys():
            # sample mean
            prototype[cls] /= self.count_by_class[cls]
            # normalization so that self.W.data is of the sampe scale as prototype_cls_norm
            prototype_cls_norm = torch.norm(prototype[cls]).clamp(min=1e-12)
            prototype[cls] = torch.div(prototype[cls], prototype_cls_norm)

            # reweight it for aggregartion
            prototype[cls] *= self.count_by_class[cls]

        self.model.return_embedding = True

        to_share = {'scaled_prototype': prototype, 'count_by_class_full': self.count_by_class_full}
        return to_share

    def _estimate_prototype_adv(self):
        """
        Estimates prototype vectors using advanced aggregation method.

        Returns:
        - to_share (dict): Dictionary containing adv_agg_prototype and count_by_class_full.
          - adv_agg_prototype (torch.Tensor): Advanced aggregated prototype vectors.
          - count_by_class_full (torch.Tensor): Full count of samples per class.
        """
        self.model.eval()
        self.model.return_embedding = True
        embeddings = []
        labels = []
        weights = []
        prototype = torch.zeros_like(self.model.prototype)
        with torch.no_grad():
            for i, (x, y) in enumerate(self.trainloader):
                # forward pass
                x, y = x.to(self.device), y.to(self.device)
                # feature_embedding is normalized
                # use the latest prototype
                feature_embedding, logits = self.model.forward(x)
                prob_ = F.softmax(logits, dim=1)
                prob = torch.gather(prob_, dim=1, index=y.view(-1, 1))
                labels.append(y)
                weights.append(prob)
                embeddings.append(feature_embedding)
        self.model.return_embedding = True
        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)
        weights = torch.cat(weights, dim=0).view(-1, 1)
        for cls in self.count_by_class.keys():
            mask = (labels == cls)
            weights_in_cls = weights[mask, :]
            feature_embedding_in_cls = embeddings[mask, :]
            prototype[cls] = torch.sum(feature_embedding_in_cls * weights_in_cls, dim=0) / torch.sum(weights_in_cls)
            prototype_cls_norm = torch.norm(prototype[cls]).clamp(min=1e-12)
            prototype[cls] = torch.div(prototype[cls], prototype_cls_norm)

        # calculate predictive power
        to_share = {'adv_agg_prototype': prototype, 'count_by_class_full': self.count_by_class_full}
        return to_share

    def upload(self):
        """
        Uploads model state dictionary and prototype estimates.

        Returns:
        - tuple: Tuple containing new_state_dict and to_share dictionary.
        - new_state_dict (dict): Updated model state dictionary.
        - to_share (dict): Prototype estimation dictionary.
        """
        if self.client_config['FedNH_client_adv_prototype_agg']:
            return self.new_state_dict, self._estimate_prototype_adv()
        else:
            return self.new_state_dict, self._estimate_prototype()
        
    def get_embeddings_and_labels(self):
        """
        Gets embeddings and labels of training data.

        Returns:
        - embeddings (torch.Tensor): Embeddings of the data.
        - labels (torch.Tensor): Labels of the data.
        """
        self.model.eval()
        self.model.return_embedding = True
        embeddings = []
        labels = []
        with torch.no_grad():
            for i, (x, y) in enumerate(self.trainloader):
                x, y = x.to(self.device), y.to(self.device)
                feature_embedding, _ = self.model.forward(x)
                embeddings.append(feature_embedding)
                labels.append(y)
        self.model.return_embedding = False
        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)
        return embeddings, labels


class FedNHServer(FedUHServer):
    """Server class for Federated NH (FedNH) algorithm."""

    def __init__(self, server_config, clients_dict, exclude, **kwargs):
        """
        Initializes a FedNH server.

        Args:
        - server_config (dict): Configuration dictionary for server parameters.
        - clients_dict (dict): Dictionary of clients participating in federated learning.
        - exclude (list): List of keys to exclude from aggregation.
        - **kwargs: Additional keyword arguments.
        """
        super().__init__(server_config, clients_dict, exclude, **kwargs)
        self.visualization_dir = self.create_simulation_directory()
        self.visualization_rounds = [1, 50, 150, 200]

    def create_simulation_directory(self):
        """
        Creates a new simulation directory under the 'visualization' folder.

        Returns:
        - str: Path to the new simulation directory.
        """
        base_dir = os.path.join(os.getcwd(), 'visualization')
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        # Find the next simulation number
        existing_sim_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        simulation_numbers = [int(d[4:]) for d in existing_sim_dirs if d.startswith('simu') and d[4:].isdigit()]
        next_simulation_number = max(simulation_numbers, default=0) + 1

        # Create the new simulation directory
        new_simulation_dir = os.path.join(base_dir, f'simu{next_simulation_number}')
        os.makedirs(new_simulation_dir)

        print(f"Simulation directory created: {new_simulation_dir}")
        return new_simulation_dir

    def tsne_visualization(self, prototypes, labels, round_num, tag):
        """
        Visualizes class prototypes using t-SNE.

        Args:
        - prototypes (torch.Tensor): Prototype vectors to visualize.
        - labels (list): List of class labels for prototypes.
        - round_num (int): Current round number for visualization.
        - tag (str): Tag to differentiate the visualization files (e.g., 'initial' or 'final').
        """
        num_samples = prototypes.shape[0]
        perplexity = min(30, num_samples - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        tsne_results = tsne.fit_transform(prototypes.cpu().numpy())

        # Color-coded by class with alpha transparency
        plt.figure(figsize=(10, 8))
        for label in set(labels):
            mask = np.array(labels) == label
            plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1], alpha=0.5, label=f'Class {label}')
        plt.title('t-SNE of Class Prototypes')
        plt.legend()
        visualization_path = os.path.join(self.visualization_dir, f"tsne_round_{round_num}_{tag}.png")
        plt.savefig(visualization_path)
        plt.close()
        print(f"t-SNE with alpha transparency saved to {visualization_path}")

        # Interactive plot using Plotly
        fig = px.scatter(x=tsne_results[:, 0], y=tsne_results[:, 1], color=labels, title='t-SNE of Class Prototypes')
        visualization_path = os.path.join(self.visualization_dir, f"tsne_round_{round_num}_{tag}_interactive.html")
        fig.write_html(visualization_path)
        print(f"Interactive t-SNE plot saved to {visualization_path}")

        # Cluster boundaries using Convex Hulls
        plt.figure(figsize=(10, 8))
        for label in set(labels):
            mask = np.array(labels) == label
            points = tsne_results[mask]
            if len(points) >= 3:
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
            plt.scatter(points[:, 0], points[:, 1], alpha=0.5, label=f'Class {label}')
        plt.title('t-SNE of Class Prototypes with Convex Hulls')
        plt.legend()
        visualization_path = os.path.join(self.visualization_dir, f"tsne_round_{round_num}_{tag}_hulls.png")
        plt.savefig(visualization_path)
        plt.close()
        print(f"t-SNE with convex hulls saved to {visualization_path}")

    def client_distribution_visualization(self, client_uploads, round_num):
        """
        Visualizes class distribution across clients.

        Args:
        - client_uploads (list): List of client uploads containing prototype estimates.
        - round_num (int): Current round number for visualization.
        """
        num_classes = self.server_config['num_classes']
        client_counts = [upload[1]['count_by_class_full'].cpu().numpy() for upload in client_uploads if 'count_by_class_full' in upload[1]]
        
        plt.figure(figsize=(10, 8))
        for i, client_count in enumerate(client_counts):
            plt.bar(range(num_classes), client_count, alpha=0.5, label=f'Class {i}')
        
        plt.xlabel('Client')
        plt.ylabel('Number of Samples')
        plt.title('Classes Distribution Across Classes')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        visualization_path = os.path.join(self.visualization_dir, f"client_distribution_round_{round_num}.png")

        try:
            plt.savefig(visualization_path)
            plt.close()
            print(f"Client distribution visualization saved to {visualization_path}")

            # Verify the file has been created
            if os.path.exists(visualization_path):
                print(f"Verified that the file {visualization_path} exists.")
            else:
                print(f"Error: The file {visualization_path} does not exist after saving.")
        except Exception as e:
            print(f"Failed to save client distribution visualization: {e}")

    def prototype_weight_visualization(self, prototype_weights, round_num, tag=''):
        """
        Visualizes the prototype weights for each class.

        Args:
        - prototype_weights (torch.Tensor): Weights of the prototypes for each class.
        - round_num (int): Current round number for visualization.
        - tag (str): Tag to differentiate the visualization files (e.g., 'pre_norm' or 'post_norm').
        """
        num_classes = prototype_weights.shape[0]
        
        plt.figure(figsize=(10, 8))
        plt.bar(range(num_classes), prototype_weights.cpu().numpy(), alpha=0.7)
        
        plt.xlabel('Class')
        plt.ylabel('Prototype Weight')
        plt.title(f'Prototype Weights for Each Class ({tag})')
        visualization_path = os.path.join(self.visualization_dir, f"prototype_weights_round_{round_num}_{tag}.png")

        try:
            plt.savefig(visualization_path)
            plt.close()
            print(f"Prototype weights visualization saved to {visualization_path}")

            # Verify the file has been created
            if os.path.exists(visualization_path):
                print(f"Verified that the file {visualization_path} exists.")
            else:
                print(f"Error: The file {visualization_path} does not exist after saving.")
        except Exception as e:
            print(f"Failed to save prototype weights visualization: {e}")

    def calculate_cluster_density(self, embeddings, labels):
        """
        Calculate cluster density for each class.

        Args:
        - embeddings (np.ndarray): Embeddings of the data points.
        - labels (np.ndarray): Labels corresponding to the embeddings.

        Returns:
        - class_densities (dict): Dictionary containing class indices and their corresponding densities.
        """
        class_densities = {}
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            if np.sum(mask) > 1:
                points = embeddings[mask]
                nbrs = NearestNeighbors(n_neighbors=2).fit(points)
                distances, _ = nbrs.kneighbors(points)
                mean_distance = np.mean(distances[:, 1])  # Mean distance to the nearest neighbor
                density = 1 / mean_distance if mean_distance > 0 else 0  # Inverse mean distance as density measure
            else:
                density = 0  # If only one point, density is 0
            class_densities[label] = density
        return class_densities

    def visualize_cluster_density(self, embeddings, labels, round_num):
        """
        Visualize cluster density using a bar plot.

        Args:
        - embeddings (torch.Tensor): Embeddings of the data points.
        - labels (torch.Tensor): Labels corresponding to the embeddings.
        - round_num (int): Current round number for visualization.
        """
        embeddings = embeddings.cpu().numpy()
        labels = labels.cpu().numpy()
        class_densities = self.calculate_cluster_density(embeddings, labels)

        plt.figure(figsize=(10, 8))
        plt.bar(class_densities.keys(), class_densities.values(), alpha=0.7)
        plt.xlabel('Class')
        plt.ylabel('Cluster Density')
        plt.title(f'Cluster Density for Each Class (Round {round_num})')
        visualization_path = os.path.join(self.visualization_dir, f"cluster_density_round_{round_num}.png")

        try:
            plt.savefig(visualization_path)
            plt.close()
            print(f"Cluster density visualization saved to {visualization_path}")

            # Verify the file has been created
            if os.path.exists(visualization_path):
                print(f"Verified that the file {visualization_path} exists.")
            else:
                print(f"Error: The file {visualization_path} does not exist after saving.")
        except Exception as e:
            print(f"Failed to save cluster density visualization: {e}")

    def visualize_client_data_clusters(self, client_uploads, round_num, num_points_per_class=100):
        """
        Visualizes client data clusters using t-SNE, excluding class prototypes.

        Args:
        - client_uploads (list): List of client uploads.
        - round_num (int): Current round number for viewing.
        - num_points_per_class (int): Number of points to sample per class.
        """
        all_embeddings = []
        all_labels = []
        for client in self.clients_dict.values():
            embeddings, labels = client.get_embeddings_and_labels()
            all_embeddings.append(embeddings)
            all_labels.append(labels)

        all_embeddings = torch.cat(all_embeddings, dim=0).cpu().numpy()
        all_labels = torch.cat(all_labels, dim=0).cpu().numpy()

        # Sample points for better readability
        sampled_embeddings = []
        sampled_labels = []
        unique_labels = np.unique(all_labels)
        for label in unique_labels:
            label_indices = np.where(all_labels == label)[0]
            sampled_indices = np.random.choice(label_indices, min(num_points_per_class, len(label_indices)), replace=False)
            sampled_embeddings.append(all_embeddings[sampled_indices])
            sampled_labels.append(all_labels[sampled_indices])

        sampled_embeddings = np.vstack(sampled_embeddings)
        sampled_labels = np.hstack(sampled_labels)

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_results = tsne.fit_transform(sampled_embeddings)

        plt.figure(figsize=(10, 8))
        for label in set(sampled_labels):
            mask = sampled_labels == label
            plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1], alpha=0.5, label=f'Class {label}')
        plt.title('t-SNE of Client Data Clusters')
        plt.legend()
        visualization_path = os.path.join(self.visualization_dir, f"client_data_clusters_round_{round_num}.png")
        plt.savefig(visualization_path)
        plt.close()
        print(f"Client data clusters t-SNE saved to {visualization_path}")

        # Interactive plot using Plotly
        fig = px.scatter(x=tsne_results[:, 0], y=tsne_results[:, 1], color=sampled_labels, title='t-SNE of Client Data Clusters')
        visualization_path = os.path.join(self.visualization_dir, f"client_data_clusters_round_{round_num}_interactive.html")
        fig.write_html(visualization_path)
        print(f"Interactive t-SNE of client data clusters saved to {visualization_path}")

        # Visualize cluster density
        self.visualize_cluster_density(torch.tensor(sampled_embeddings), torch.tensor(sampled_labels), round_num)


            
    def aggregate(self, client_uploads, round):
        """
        Aggregates updates from clients to update the server model.

        Args:
        - client_uploads (list): A list of tuples containing the state dictionaries and prototype dictionaries from clients.
        - round (int): The current round number of the federated learning process.
        """
        # Calculate the learning rate for the server for the current round
        server_lr = self.server_config['learning_rate'] * (self.server_config['lr_decay_per_round'] ** (round - 1))
        
        # Determine the number of participants
        num_participants = len(client_uploads)
        
        # Initialize variables for aggregation
        update_direction_state_dict = None
        cumsum_per_class = torch.zeros(self.server_config['num_classes']).to(self.clients_dict[0].device)
        agg_weights_vec_dict = {}

        # Clone the initial prototypes from the server model state dictionary
        initial_prototypes = self.server_model_state_dict['prototype'].clone()
        self.tsne_visualization(initial_prototypes, list(range(self.server_config['num_classes'])), round, tag='initial')

        with torch.no_grad():
            # Iterate through client uploads to process and aggregate updates
            for idx, (client_state_dict, prototype_dict) in enumerate(client_uploads):
                if self.server_config['FedNH_server_adv_prototype_agg']:
                    # Check for the presence of 'adv_agg_prototype' in the prototype dictionary
                    if 'adv_agg_prototype' in prototype_dict:
                        mu = prototype_dict['adv_agg_prototype']
                        W = self.server_model_state_dict['prototype']
                        agg_weights_vec_dict[idx] = torch.exp(torch.sum(W * mu, dim=1, keepdim=True))
                    else:
                        raise KeyError(f"Client {idx} did not provide 'adv_agg_prototype'. Check client configuration.")
                else:
                    # Check for the presence of 'scaled_prototype' in the prototype dictionary
                    if 'scaled_prototype' in prototype_dict:
                        cumsum_per_class += prototype_dict['count_by_class_full']
                    else:
                        raise KeyError(f"Client {idx} did not provide 'scaled_prototype'. Check client configuration.")
                
                # Calculate the client update by subtracting the server model state dictionary from the client state dictionary
                client_update = linear_combination_state_dict(client_state_dict,
                                                            self.server_model_state_dict,
                                                            1.0,
                                                            -1.0,
                                                            exclude=self.exclude_layer_keys)
                if idx == 0:
                    update_direction_state_dict = client_update
                else:
                    update_direction_state_dict = linear_combination_state_dict(update_direction_state_dict,
                                                                                client_update,
                                                                                1.0,
                                                                                1.0,
                                                                                exclude=self.exclude_layer_keys)

            # Update the server model state dictionary using the aggregated updates and learning rate
            self.server_model_state_dict = linear_combination_state_dict(self.server_model_state_dict,
                                                                        update_direction_state_dict,
                                                                        1.0,
                                                                        server_lr / num_participants,
                                                                        exclude=self.exclude_layer_keys)

            avg_prototype = torch.zeros_like(self.server_model_state_dict['prototype'])
            if self.server_config['FedNH_server_adv_prototype_agg']:
                m = self.server_model_state_dict['prototype'].shape[0]
                sum_of_weights = torch.zeros((m, 1)).to(avg_prototype.device)
                for idx, (_, prototype_dict) in enumerate(client_uploads):
                    sum_of_weights += agg_weights_vec_dict[idx]
                    avg_prototype += agg_weights_vec_dict[idx] * prototype_dict['adv_agg_prototype']
                avg_prototype /= sum_of_weights
            else:
                for _, prototype_dict in client_uploads:
                    avg_prototype += prototype_dict['scaled_prototype'] / cumsum_per_class.view(-1, 1)

            # Visualize prototype weights before normalization
            pre_norm_weights = avg_prototype.norm(dim=1)
            self.prototype_weight_visualization(pre_norm_weights, round, tag='pre_norm')

            # Normalize the average prototype
            avg_prototype = F.normalize(avg_prototype, dim=1)
            weight = self.server_config['FedNH_smoothing']
            temp = weight * self.server_model_state_dict['prototype'] + (1 - weight) * avg_prototype
            self.server_model_state_dict['prototype'].copy_(F.normalize(temp, dim=1))

            # Visualize prototype weights after normalization
            post_norm_weights = self.server_model_state_dict['prototype'].norm(dim=1)
            self.prototype_weight_visualization(post_norm_weights, round, tag='post_norm')

            # Visualize t-SNE representation of the final prototypes
            self.tsne_visualization(temp, list(range(self.server_config['num_classes'])), round, tag='final')

            # Visualize client distribution and data clusters
            self.client_distribution_visualization(client_uploads, round)
            self.visualize_client_data_clusters(client_uploads, round)
                        

