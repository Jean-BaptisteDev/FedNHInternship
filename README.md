# FedNH

This repo provides an implementation of `FedNH` proposed in for [Tackling Data Heterogeneity in Federated Learning with Class Prototypes](https://arxiv.org/abs/2212.02758), which is accepted by AAAI2023. In companion, we also provide our implementation of benchmark algorithms.

## Prepare Dataset

Please create a folder `data` under the root directory.

```
mkdir ~/data
```

* Cifar10, Cifar100: No extra steps are required.

* TinyImageNet
 * Download the dataset `cd ~/data && wget http://cs231n.stanford.edu/tiny-imagenet-200.zip`
 * Unzip the file `unzip tiny-imagenet-200.zip`

## Run scripts
We prepared a python file `/experiments/gen_script.py` to generate bash commands to run experiments.

To reproduce the results for Cifar10/Cifar100, just set the variable `purpose` to `Cifar` in the `gen_script.py` file. Similarly, set `purpose` to `TinyImageNet` to run experiments for TinyImageNet.

`gen_script.py` will create a set of bash files named as `[method]_dir.sh`. Then use, for example, `bash FedAvg.sh` to run experiments.

We include a set of bash files to run experiments on `Cifar` in this submission.

## Organization of the code
The core code can be found at `src/flbase/`. Our framework builds upon three abstrac classes `server`, `clients`, and `model`. And their concrete implementations can be found in `models` directory and the `startegies` directory, respectively.

* `src/flbase/models`: We implemented or borrowed the implementation of (1) Convolution Neural Network and (2) Resnet18.
* `src/flbase/strategies`: We implement `CReFF`, `Ditto`, `FedAvg`, `FedBABU`, `FedNH`, `FedPer`, `FedProto`, `FedRep`, `FedROD`. Each file provides the concrete implementation of the corresponding `server` class and `client` class.

Helper functions, for example, generating non-iid data partition, can be found in `src/utils.py`.


## Credits
The code base is developed with extensive references to the following GitHub repos. Some code snippets are directly taken from the original implementation.

1. FedBABU: https://github.com/jhoon-oh/FedBABU
2. CReFF: https://github.com/shangxinyi/CReFF-FL
3. FedROD: https://openreview.net/revisions?id=I1hQbx10Kxn
4. Personalized Federated Learning Platform: https://github.com/TsingZ0/PFL-Non-IID
5. FedProxL: https://github.com/litian96/FedProx
6. NIID-Bench: https://github.com/Xtra-Computing/NIID-Bench
7. FedProto: https://github.com/yuetan031/fedproto