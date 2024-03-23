from argparse import Namespace
from avalanche.benchmarks import GenericCLScenario
from typing import Any, List, Tuple
import os
import csv
import torch
from Source.train_eval import test
from Source.helper import get_data_loaders, get_device, reduce_or_flat_convs
from Source.resnet18 import ResNet18
from torch.utils.data import DataLoader
import pickle
from torchsummary import summary


def group_list_using_l1(l1, l2):
    groups_1 = []
    groups_2 = []
    for item_1, item_2 in zip(l1, l2):
        if not groups_1 or item_1 != groups_1[-1][-1]:
            groups_1.append([item_1])
            groups_2.append([item_2])
        else:
            groups_1[-1].append(item_1)
            groups_2[-1].append(item_2)

    return groups_1, groups_2


def dataset2input(dataset_name):
    if dataset_name == "MNIST" or dataset_name == "FMNIST" or dataset_name == "EMNIST":
        return (1, 28, 28)
    if dataset_name == "CIFAR10" or dataset_name == "CIFAR100":
        return (3, 32, 32)
    else:
        return (3, 64, 64)


def calculate_accuracy(preds, gts):
    if len(preds) != len(gts):
        raise ValueError("The length of preds and gts should be equal")
    correct_predictions = sum(p1 == p2 for p1, p2 in zip(preds, gts))
    accuracy = correct_predictions / len(gts)
    return accuracy


def acc_prev_tasks(args: Namespace, context_detector: Any, task_index: int,
                   scenario: GenericCLScenario, network: Any, til_eval=False) -> Tuple[List, List, List, List, List, List, List]:
    all_accuracies, all_preds, all_trues = [], [], []
    all_episode_preds, all_episode_preds_train, all_episode_gts, all_episode_train_gts = [], [], [], []
    for episode_id, (train_task, val_task, test_task) in enumerate(zip(scenario.train_stream[:task_index],
                                                                       scenario.val_stream[:task_index],  # type: ignore
                                                                       scenario.test_stream[:task_index]), 1):
        episode_to_model = episode_id if til_eval else None
        task_classes = str(test_task.classes_in_this_experience)
        train_loader, _, test_loader = get_data_loaders(args, train_task, val_task, test_task)
        val_acc = 0.0
        train_acc, _, _, episode_preds_train = test(network, context_detector,
                                                    train_loader, episode_to_model, return_preds=True)  # type: ignore
        test_acc, preds, ground_truths, episode_preds = test(network, context_detector, test_loader,
                                                             episode_to_model, return_preds=True)  # type: ignore
        all_preds.append(preds)
        all_trues.append(ground_truths)
        all_episode_preds.extend(episode_preds)
        all_episode_preds_train.extend(episode_preds_train)
        all_episode_train_gts.extend([episode_id]*len(episode_preds_train))
        all_episode_gts.extend([episode_id]*len(episode_preds))
        all_accuracies.append((task_classes, [train_acc, val_acc, test_acc]))
    return all_accuracies, all_preds, all_trues, all_episode_preds, all_episode_preds_train,  all_episode_train_gts, all_episode_gts


def log_end_of_episode(args: Namespace, network: Any, context_detector: Any, scenario: GenericCLScenario, episode_index: int, dirpath: str):
    dirpath = os.path.join(dirpath, "Episode_{}".format(episode_index))
    csvfile = open(os.path.join(dirpath, "Episode_{}.csv".format(episode_index)), 'w', newline='')
    writer = csv.writer(csvfile)
    writer = write_units(writer, network)
    # TIL
    prev_task_accs, _, _, _, _, _, _ = acc_prev_tasks(args, context_detector, episode_index, scenario, network, til_eval=True)
    writer.writerow(["Task Incremental Learning"])
    for task_classes, (train_acc, val_acc, test_acc) in prev_task_accs:
        writer.writerow([str(task_classes), "Train Acc: {:.2f}".format(train_acc),
                         "Val Acc: {:.2f}".format(val_acc), "Test Acc: {:.2f}".format(test_acc)])

    # CIL
    prev_task_accs, test_predictions, test_gts, all_episode_preds, all_episode_preds_train, all_episode_train_gts, all_episode_gts = acc_prev_tasks(
        args, context_detector, episode_index, scenario, network)
    writer.writerow(["Class Incremental Learning"])
    for task_classes, (train_acc, val_acc, test_acc) in prev_task_accs:
        writer.writerow([str(task_classes), "Train Acc: {:.2f}".format(train_acc),
                        "Val Acc: {:.2f}".format(val_acc), "Test Acc: {:.2f}".format(test_acc)])

    writer.writerow(["Episode-id Prediction Accuracy on Test Set: {:.2f}".format(calculate_accuracy(all_episode_preds,
                                                                                                    all_episode_gts))])
    writer.writerow(["Episode-id Prediction Accuracy on Train Set: {:.2f}".format(calculate_accuracy(all_episode_preds_train,
                                                                                                     all_episode_train_gts))])

    # Write all accuracies
    writer.writerow(["Previous Task Test Accuracies"] + [round((preds == gts).mean(), 4)
                    for preds, gts in zip(test_predictions, test_gts)])
    csvfile.close()

    # Save activations and labels uncomment if needed
    # test_dataset_activations = []
    # test_dataset_labels = []
    # for test_episode in scenario.test_stream[:episode_index]:
    #     test_loader = DataLoader(test_episode.dataset, batch_size=args.batch_size, shuffle=False)
    #     network.eval()
    #     with torch.no_grad():
    #         for data, target, _ in test_loader:
    #             data = data.to(get_device())
    #             target = target.to(get_device())
    #             _, output = reduce_or_flat_convs(network.get_activations(data))
    #             test_dataset_activations.append(output)
    #             test_dataset_labels.append(target)
    # with open(os.path.join(dirpath, 'test_dataset_activations.pkl'), 'wb') as f:
    #     pickle.dump(test_dataset_activations, f)
    # with open(os.path.join(dirpath, 'test_dataset_labels.pkl'), 'wb') as f:
    #     pickle.dump(test_dataset_labels, f)
    # with open(os.path.join(dirpath, 'context_detector.pkl'), 'wb') as f:
    #     pickle.dump(context_detector, f)


def log_end_of_sequence(args: Namespace, network: Any, context_detector: Any, scenario: GenericCLScenario, dirpath: str):
    csvfile = open(os.path.join(dirpath, "End_of_Sequence.csv"), 'w', newline='')
    writer = csv.writer(csvfile)
    writer = write_units(writer, network)
    prev_task_accs, _, _, _, _, _, _ = acc_prev_tasks(args, context_detector,
                                                      args.number_of_tasks, scenario, network)
    for task_classes, (train_acc, val_acc, test_acc) in prev_task_accs:
        writer.writerow([str(task_classes), "Train Acc: {:.2f}".format(train_acc),
                         "Val Acc: {:.2f}".format(val_acc), "Test Acc: {:.2f}".format(test_acc)])
    csvfile.close()

    possible_params, actual_params, sparsity = network.compute_weight_sparsity_2()
    summary_str = str(summary(network, dataset2input(args.dataset), verbose=0))
    with open(os.path.join(dirpath, "model_summary.txt"), 'w', newline='') as file:
        file.write("Possible Parameters: {}  Actual Parameters: {} Sparsity: {} \n".format(
            possible_params, actual_params, sparsity))
        file.write(summary_str)


def log_end_of_phase(args: Namespace, network: Any, context_detector: Any, episode_index: int, phase_index: int,
                     train_loader: Any, val_loader: Any, test_loader: Any, dirpath: str):
    dirpath = os.path.join(dirpath, "Episode_{}".format(episode_index), "Phase_{}".format(phase_index))
    os.makedirs(dirpath)
    csvfile = open(os.path.join(dirpath, "Phase_{}.csv".format(phase_index)), 'w', newline='')
    writer = csv.writer(csvfile)
    writer = write_units(writer, network)
    writer.writerow(["Test Accuracy", test(network, context_detector, test_loader)])
    csvfile.close()


def write_units(writer, network: Any):
    if isinstance(network, ResNet18):
        all_units = [list(range(a)) for a, _ in network.layers]
    else:
        weights = network.get_weight_bias_masks_numpy()
        all_units = [list(range(weights[0][0].shape[1]))] + [list(range(w[1].shape[0])) for w in weights]
    writer.writerow(["All Units"] + [len(u) for u in all_units])
    writer.writerow(["Young Neurons"] + [len((u == 0).nonzero()[0])
                    for u, _ in network.unit_ranks])
    writer.writerow(["Learner Neurons"] + [len((u == 1).nonzero()[0])
                    for u, _ in network.unit_ranks])
    writer.writerow(["Mature Neurons"] + [len((u > 1).nonzero()[0])
                    for u, _ in network.unit_ranks])
    return writer
