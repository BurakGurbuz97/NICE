#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate research


# MNIST Standard Memory Seed 0
python launch.py --experiment_name "MNIST_MEMO1_SEED0" --memo_per_class_context "50" --context_layers 0 1 2 3 4 \
       --context_learner "LogisticRegression(random_state=0, max_iter=50, C=0.4)" --dataset "MNIST" --number_of_tasks "5" \
       --model "CNN_MNIST" --seed "0" --learning_rate "0.01" \
       --batch_size "32" --weight_decay "0.0" --phase_epochs "5" --activation_perc "95.0" --max_phases "5"

# FashionMNIST Standard Memory Seed 0
python launch.py --experiment_name "FashionMNIST_MEMO1_SEED0" --memo_per_class_context "50" --context_layers 0 1 2 3 4 \
       --context_learner "LogisticRegression(random_state=0, max_iter=50, C=0.4)" --dataset "FMNIST" --number_of_tasks "5" \
       --model "CNN_MNIST" --seed "0" --learning_rate "0.005" \
       --batch_size "32" --weight_decay "0.0" --phase_epochs "5" --activation_perc "95.0" --max_phases "5"

# EMNIST Standard Memory Seed 0
python launch.py --experiment_name "EMNIST_MEMO1_SEED0" --memo_per_class_context "50" --context_layers 0 1 2 3 4 \
       --context_learner "LogisticRegression(random_state=0, max_iter=20, C=0.15)" --dataset "EMNIST" --number_of_tasks "13" \
       --model "CNN_MNIST" --seed "0" --learning_rate "0.005" --batch_size "32" --weight_decay "0.0" --phase_epochs "5" \
       --activation_perc "95.0" --max_phases "5"

# CIFAR10 Standard Memory Seed 0
python launch.py --experiment_name "CIFAR10_MEMO1_SEED0" --memo_per_class_context "150" \
       --context_layers 0 1 2 3 4 5 6 7 8 9 10 11 --context_learner "LogisticRegression(random_state=0, max_iter=20, C=0.01)" \
       --dataset "CIFAR10" --number_of_tasks "5" --model "VGG11_SLIM" --seed "0" --learning_rate "0.01" --batch_size "32" \
       --weight_decay "0.0001"  --phase_epochs "10" --activation_perc "95.0" --max_phases "5"


# CIFAR100 Standard Memory Seed 0
python launch.py --experiment_name "CIFAR100_MEMO1_SEED0" --memo_per_class_context "50" \
       --context_layers 0 1 2 3 4 5 6 7 8 9 10 11 --context_learner "LogisticRegression(random_state=0, max_iter=20, C=0.005)" \
       --dataset "CIFAR100" --number_of_tasks "10" --model "VGG11_SLIM" --seed "0" --learning_rate "0.005" --batch_size "32" \
       --weight_decay "0.001"  --phase_epochs "15" --activation_perc "95.0" --max_phases "5"


# TinyImageNet Standard Memory Seed 0
python launch.py --experiment_name "TinyImagenet_MEMO1_SEED0" --memo_per_class_context "25" \
       --context_layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 \
       --context_learner "LogisticRegression(random_state=0, max_iter=25, C=0.001)" --dataset "TinyImagenet" \
       --number_of_tasks "5" --model "ResNet18" --seed "0" --learning_rate "0.01" --batch_size "64" --weight_decay "0.0" \
       --phase_epochs "20" --activation_perc "97.5" --max_phases "5"