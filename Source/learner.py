from argparse import Namespace
from avalanche.benchmarks import GenericCLScenario, TCLExperience
from typing import Dict, Any
import torch
from Source.helper import random_prune, get_device, get_data_loaders
from Source.context_detector import ContextDetector
from Source.nice_operations import increase_unit_ranks, update_freeze_masks, select_learner_units
from Source.nice_operations import drop_young_to_learner, grow_all_to_young
from Source.log import log_end_of_episode, log_end_of_sequence, log_end_of_phase
from Source.train_eval import test, phase_training_ce


class Learner():

    def __init__(self, args: Namespace, network: Any, scenario: GenericCLScenario,
                 input_size: int, task2classes: Dict, log_dirpath: str):
        self.args, self.input_size = args, input_size
        self.task2classes = task2classes
        self.log_dirpath = log_dirpath
        self.optim_obj = getattr(torch.optim, args.optimizer)
        # this is needed to assign masks for later
        self.network = random_prune(network.to(get_device()), 0.0)
        self.context_detector = ContextDetector(args, network.penultimate_layer_size, task2classes)
        self.original_scenario = scenario
        print("Model: \n", self.network)

    def start_episode(self, train_episode: TCLExperience, val_episode: TCLExperience, test_episode: TCLExperience, episode_index: int):
        print("****** Starting Episode-{}   Classes: {} ******".format(episode_index, train_episode.classes_in_this_experience))

    def end_episode(self, train_episode: TCLExperience, val_episode: TCLExperience, test_episode: TCLExperience, episode_index: int):
        print("******  Ending Episode ****** ")
        self.context_detector.push_activations(self.network, train_episode, episode_index)
        self.network = increase_unit_ranks(self.network)
        self.network = update_freeze_masks(self.network)
        self.network.freeze_bn_layers()
        log_end_of_episode(self.args, self.network, self.context_detector,
                           self.original_scenario, episode_index, self.log_dirpath)

    def learn_episode(self, train_episode: TCLExperience, val_episode: TCLExperience, test_episode: TCLExperience, episode_index: int):
        train_loader, val_loader, test_loader = get_data_loaders(self.args, train_episode, val_episode, test_episode, episode_index)
        phase_index = 1
        selection_perc = 100.0
        loss = torch.nn.CrossEntropyLoss()
        while (True):
            print('Selecting Units (Ratio: {})'.format(selection_perc))
            self.network = select_learner_units(self.network, selection_perc, train_episode, episode_index)
            print('Dropping connections')
            self.network = drop_young_to_learner(self.network)
            print('Fixing Young connections')
            self.network = grow_all_to_young(self.network)
            print("Sparsity phase-{}: {:.2f}".format(phase_index,self.network.compute_weight_sparsity()))

            if self.args.optimizer == "SGD":
                optimizer = self.optim_obj(self.network.parameters(), lr=self.args.learning_rate, weight_decay=0.0,
                                           momentum=self.args.sgd_momentum)
            else:
                optimizer = self.optim_obj(self.network.parameters(), lr=self.args.learning_rate, weight_decay=0.0)

            self.network = phase_training_ce(self.network, self.args.phase_epochs, loss, optimizer, train_loader, self.args)

            # Push context_detector
            self.context_detector.push_activations(self.network, train_episode, episode_index)

            # Compute validation accuracy
            test_accuracy = test(self.network, self.context_detector, test_loader)

            print("Episode-{}, Phase-{}, Episode Test Class Accuracy: {}".format(episode_index, phase_index, test_accuracy))

            log_end_of_phase(self.args, self.network, self.context_detector, episode_index, phase_index,
                             train_loader, val_loader, test_loader, self.log_dirpath)
            phase_index = phase_index + 1
            selection_perc = self.args.activation_perc

            # Use all neurons in the last episode, selection_perc = 100
            if episode_index == self.args.number_of_tasks:
                selection_perc = 100.0
            if phase_index > self.args.max_phases:
                break

    def learn_all_episodes(self):
        for episode_index, (train_task, val_task, test_task) in enumerate(zip(self.original_scenario.train_stream,
                                                                              self.original_scenario.val_stream,  # type: ignore
                                                                              self.original_scenario.test_stream), 1):
            self.start_episode(train_task, val_task, test_task, episode_index)
            self.learn_episode(train_task, val_task, test_task, episode_index)
            self.end_episode(train_task, val_task, test_task, episode_index)
        log_end_of_sequence(self.args, self.network, self.context_detector,
                            self.original_scenario, self.log_dirpath)
