import os
import platform
import glob

def gen_command(
        purpose,
        device,
        global_seed,
        use_wandb,
        yamlfile,
        strategy,
        num_clients,
        participate_ratio,
        partition,
        partition_arg,
        partition_val,
        num_rounds,
        client_lr,
        client_lr_scheduler,
        sgd_momentum,
        sgd_weight_decay,
        num_epochs,
        **kwargs):

    command = f"python3 ../main.py  --purpose {purpose} --device {device} --global_seed {global_seed} --use_wandb {use_wandb} --yamlfile {yamlfile} --strategy {strategy} --num_clients {num_clients} --participate_ratio {participate_ratio} --partition {partition} --{partition_arg} {partition_val} --num_rounds {num_rounds} --client_lr {client_lr} --client_lr_scheduler {client_lr_scheduler} --sgd_momentum {sgd_momentum} --sgd_weight_decay {sgd_weight_decay} --num_epochs {num_epochs}"

    command_hyper = ""
    for k, v in kwargs.items():
        command_hyper += f" --{k} {v}"
    command += command_hyper + " &"
    return command

if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")

    # Delete .sh files in a cross-platform way
    for filename in glob.glob('*.sh'):
        try:
            os.remove(filename)
            print(f"Deleted file: {filename}")
        except OSError as e:
            print(f"Error deleting file {filename}: {e}")

    # purpose = 'TinyImageNet'
    purpose = 'Cifar'
    num_gpu = 4
    if purpose == 'TinyImageNet':
        strategy_hyper = [('FedAvg', {'no_norm': False}), ('FedNH', {'no_norm': True}), ('FedNHPlus', {'no_norm': True}),
                          ('FedROD', {'FedROD_hyper_clf': False, 'FedROD_phead_separate': True, 'no_norm': False}), ('FedProto', {'no_norm': False}),
                          ('FedRep', {'no_norm': False}), ('FedBABU', {'FedBABU_finetune_epoch': 5, 'no_norm': False}),
                          ('FedPer', {'no_norm': False}), ('Ditto', {'Ditto_lambda': 0.75, 'no_norm': False}),
                          ('Local', {'no_norm': False}),
                          ('CReFF', {'CReFF_lr_feature': 0.01, 'CReFF_lr_net': 0.01})
                          ]
        planned, actual = 0, 0
        yamlfile_lst = ['./TinyImageNet_ResNet.yaml']
        num_round = 200
        client_lr = 0.01
        client_lr_scheduler = 'diminishing'
        sgd_momentum = 0.9
        sgd_weight_decay = 0.001
        num_epochs = 5
        for run_type in ['beta']:
            for yamlfile in yamlfile_lst:
                for strategy, hyper in strategy_hyper:
                    if run_type == 'beta':
                        for beta in ['0.3', '1.0']:
                            for pratio in [0.1]:
                                cuda = f'cuda:{planned % num_gpu}'
                                if hyper is not None:
                                    command = gen_command(purpose, cuda,
                                                          0, False, yamlfile, strategy,
                                                          100, pratio, 'noniid-label-distribution', 'beta', beta, num_round,
                                                          client_lr, client_lr_scheduler, sgd_momentum, sgd_weight_decay, num_epochs, **hyper
                                                          )
                                else:
                                    command = gen_command(purpose, cuda,
                                                          0, False, yamlfile, strategy,
                                                          100, pratio, 'noniid-label-distribution', 'beta', beta, num_round,
                                                          client_lr, client_lr_scheduler, sgd_momentum, sgd_weight_decay, num_epochs
                                                          )

                                planned += 1
                                if command is not None:
                                    actual += 1
                                    filename = f'{strategy}_dir.sh'
                                    print(f"Writing command to {filename}")
                                    # Open file in binary mode and use LF line endings
                                    with open(filename, 'ab') as f:
                                        f.write(command.encode('utf-8') + b'\n')
                                    print(f"Created file: {filename}")

        print(f"actual/planned:{actual}/{planned}")

    if purpose == 'Cifar':
        strategy_hyper = [('FedAvg', None), ('FedNH', None), ('FedNHPlus', {'no_norm': True}),
                          ('FedROD', {'FedROD_phead_separate': True}), ('FedProto', None),
                          ('FedRep', None), ('FedBABU', {'FedBABU_finetune_epoch': 5}),
                          ('FedPer', None), ('Ditto', {'Ditto_lambda': 0.75}),
                          ('Local', {'no_norm': False}),
                          ('CReFF', {'CReFF_lr_feature': 0.01, 'CReFF_lr_net': 0.01})
                          ]
        yamlfile_lst = ['./Cifar10_Conv2Cifar.yaml', './Cifar100_Conv2Cifar.yaml']
        planned, actual = 0, 0
        num_round = 200
        client_lr = 0.01
        client_lr_scheduler = 'diminishing'
        sgd_momentum = 0.9
        sgd_weight_decay = 0.00001
        num_epochs = 5
        for run_type in ['beta']:
            for yamlfile in yamlfile_lst:
                for strategy, hyper in strategy_hyper:
                    if run_type == 'beta':
                        for beta in ['0.3', '1.0']:
                            for pratio in [0.1]:
                                cuda = f'cuda:{planned % num_gpu}'
                                if hyper is not None:
                                    command = gen_command(purpose, cuda,
                                                          0, False, yamlfile, strategy,
                                                          100, pratio, 'noniid-label-distribution', 'beta', beta, num_round,
                                                          client_lr, client_lr_scheduler, sgd_momentum, sgd_weight_decay, num_epochs, **hyper
                                                          )
                                else:
                                    command = gen_command(purpose, cuda,
                                                          0, False, yamlfile, strategy,
                                                          100, pratio, 'noniid-label-distribution', 'beta', beta, num_round,
                                                          client_lr, client_lr_scheduler, sgd_momentum, sgd_weight_decay, num_epochs
                                                          )

                                planned += 1
                                if command is not None:
                                    actual += 1
                                    filename = f'{strategy}_dir.sh'
                                    print(f"Writing command to {filename}")
                                    # Open file in binary mode and use LF line endings
                                    with open(filename, 'ab') as f:
                                        f.write(command.encode('utf-8') + b'\n')
                                    print(f"Created file: {filename}")

        print(f"actual/planned:{actual}/{planned}")
