import argparse
import os
import pickle
import shutil
import random
from typing import Tuple, Dict

from avalanche.benchmarks import GenericCLScenario
from avalanche.benchmarks.classic import SplitCIFAR10, SplitCIFAR100, SplitFMNIST, SplitMNIST, SplitTinyImageNet
#from avalanche.benchmarks.generators import benchmark_with_validation_stream, nc_benchmark
from avalanche.benchmarks import nc_benchmark, benchmark_with_validation_stream#now its 0.5.0 ver for ava...
from avalanche.benchmarks.datasets import EMNIST
import numpy as np
import torch
from torch.backends import cudnn
from torchvision import transforms


DATASET_PATH = os.path.join(os.path.abspath('..'), 'datasets_new')
LOG_PATH = os.path.join(os.path.abspath('.'), 'Logs')


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Experiment')
    # Logging params
    parser.add_argument('--experiment_name', type=str, default='MNIST')

    # Context Detector  Params
    parser.add_argument('--memo_per_class_context', type=int, default=50)
    parser.add_argument('--context_layers', type=int, nargs='+', default=[0, 1, 2, 3, 4])  # -1 = Penultimate, 0 = Input
    parser.add_argument('--context_learner', type=str, default="LogisticRegression(random_state=0, max_iter=50, C=0.4)")

    # Dataset params
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--number_of_tasks', type=int, default=5)

    # Architectural params
    parser.add_argument('--model', type=str, default='CNN_MNIST')
    parser.add_argument('--resnet_multiplier', type=float, default=1.0)

    # Learning params
    parser.add_argument('--seed', type=int,  default=0)
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--sgd_momentum', type=float, default=0.90)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    # Algortihm params
    parser.add_argument('--phase_epochs', type=int, default=5)
    parser.add_argument("--activation_perc", type=float, default=95.0)
    parser.add_argument("--max_phases", type=int, default=5)

    return parser.parse_args()


def create_log_dirs(args: argparse.Namespace) -> str:
    dirpath = os.path.join(LOG_PATH, args.experiment_name)
    # Remove existing files/dirs
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)

    # Create log dirs and save experiment args
    os.makedirs(dirpath)
    with open(os.path.join(dirpath, 'args.pkl'), 'wb') as file:
        pickle.dump(args, file)

    for task_id in range(1, args.number_of_tasks + 1):
        os.makedirs(os.path.join(dirpath, "Episode_{}".format(task_id)))

    return dirpath


def get_experience_streams(args: argparse.Namespace) -> Tuple[GenericCLScenario, int, int, Dict]:
    if args.dataset == "MNIST":
        stream = SplitMNIST(n_experiences=args.number_of_tasks,
                            seed=args.seed, dataset_root=DATASET_PATH, fixed_class_order=list(range(10)))
        stream_with_val = benchmark_with_validation_stream(stream, validation_size=0.01, output_stream="val", shuffle=True)
        task2classes = dict((index, stream.classes_in_this_experience)
                            for index, stream in enumerate(stream_with_val.train_stream, 1))
        return (stream_with_val, 784, 10, task2classes)

    if args.dataset == "FMNIST":
        stream = SplitFMNIST(n_experiences=args.number_of_tasks,
                             seed=args.seed, dataset_root=DATASET_PATH, fixed_class_order=list(range(10)))
        stream_with_val = benchmark_with_validation_stream(stream, validation_size=0.01, output_stream="val", shuffle=True)
        task2classes = dict((index, stream.classes_in_this_experience)
                            for index, stream in enumerate(stream_with_val.train_stream, 1))
        return (stream_with_val, 784, 10, task2classes)

    if args.dataset == "EMNIST":
        emnist_train = EMNIST(root=DATASET_PATH, train=True, split='letters', download=True)
        emnist_train.targets = emnist_train.targets - 1
        emnist_test = EMNIST(root=DATASET_PATH, train=False, split='letters', download=True)
        emnist_test.targets = emnist_test.targets - 1
        stream = nc_benchmark(train_dataset=emnist_train, test_dataset=emnist_test,  # type: ignore
                              n_experiences=args.number_of_tasks, task_labels=False, shuffle=False,
                              seed=args.seed, fixed_class_order=list(
                                  map(lambda x: int(x), emnist_train.targets.unique())),
                              train_transform=transforms.ToTensor(),
                              eval_transform=transforms.ToTensor())

        stream_with_val = benchmark_with_validation_stream(stream, validation_size=0.01, output_stream="val", shuffle=True)
        task2classes = dict((index, stream.classes_in_this_experience)
                            for index, stream in enumerate(stream_with_val.train_stream, 1))
        return (stream_with_val, 784, 26, task2classes)

    if args.dataset == "CIFAR10":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        stream = SplitCIFAR10(n_experiences=args.number_of_tasks,
                              seed=args.seed, dataset_root=DATASET_PATH, fixed_class_order=list(range(10)),
                              train_transform=transform, eval_transform=transform)
        stream_with_val = benchmark_with_validation_stream(stream, validation_size=0.01, output_stream="val", shuffle=True)
        task2classes = dict((index, stream.classes_in_this_experience)
                            for index, stream in enumerate(stream_with_val.train_stream, 1))
        return (stream_with_val, 3, 10, task2classes)

    if args.dataset == "CIFAR100":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        stream = SplitCIFAR100(n_experiences=args.number_of_tasks,
                               seed=args.seed, dataset_root=DATASET_PATH, fixed_class_order=list(range(100)),
                               train_transform=transform, eval_transform=transform)
        stream_with_val = benchmark_with_validation_stream(stream, validation_size=0.001, output_stream="val", shuffle=True)
        task2classes = dict((index, stream.classes_in_this_experience)
                            for index, stream in enumerate(stream_with_val.train_stream, 1))
        return (stream_with_val, 3, 100, task2classes)

    if args.dataset == "TinyImagenet":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4802, 0.4480, 0.3975],
                                 std=[0.2770, 0.2691, 0.2821])
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4802, 0.4480, 0.3975],
                                 std=[0.2770, 0.2691, 0.2821])
        ])
        stream = SplitTinyImageNet(n_experiences=args.number_of_tasks,
                                   seed=args.seed, dataset_root=DATASET_PATH, fixed_class_order=list(range(200)),
                                   train_transform=train_transform, eval_transform=test_transform)
        stream_with_aug = benchmark_with_validation_stream(stream, validation_size=0.01, output_stream="val", shuffle=True)
        task2classes = dict((index, stream.classes_in_this_experience)
                            for index, stream in enumerate(stream_with_aug.train_stream, 1))
        return (stream_with_aug, 3, 200, task2classes)  # type: ignore

    raise Exception("Dataset {} is not defined!".format(args.dataset))
