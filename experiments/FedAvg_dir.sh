python3 ../main.py  --purpose Cifar --device cuda --global_seed 0 --use_wandb True --yamlfile ./Cifar10_Conv2Cifar.yaml --strategy FedAvg --num_clients 100 --participate_ratio 0.2 --partition noniid-label-distribution --beta 0.3 --num_rounds 100 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 &
python3 ../main.py  --purpose Cifar --device cuda --global_seed 0 --use_wandb True --yamlfile ./Cifar10_Conv2Cifar.yaml --strategy FedAvg --num_clients 100 --participate_ratio 0.2 --partition noniid-label-distribution --beta 1.0 --num_rounds 100 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 &

