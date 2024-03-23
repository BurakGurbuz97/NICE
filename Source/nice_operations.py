from typing import List, Any
import numpy as np
import torch
import copy
from Source.helper import get_device
from Source.architecture import SparseConv2d, SparseLinear, SparseOutput
from Source.resnet18 import ResNet18
from torch.utils.data import DataLoader


def get_current_young_neurons(unit_ranks: List) -> List:
    current_young_neurons = [list((ranks == 0).nonzero()[0])
                             for ranks, _ in unit_ranks]
    return current_young_neurons


def get_current_learner_neurons(unit_ranks: List) -> List:
    current_learner_neurons = [list((ranks == 1).nonzero()[0])
                               for ranks, _ in unit_ranks]
    return current_learner_neurons


def increase_unit_ranks(network: Any) -> Any:
    # Updating unit_ranks
    unit_rank_list = []
    for unit_layer, name in network.unit_ranks[1:]:
        unit_layer[unit_layer != 0] = unit_layer[unit_layer != 0] + 1
        unit_rank_list.append((unit_layer, name))

    unit_rank_list = [(network.unit_ranks[0][0], network.unit_ranks[0][1])] + unit_rank_list
    network.unit_ranks = unit_rank_list
    # Updating current_young_neurons, current_learner_neurons
    network.current_young_neurons = get_current_young_neurons(network.unit_ranks)
    network.current_learner_neurons = get_current_learner_neurons(network.unit_ranks)

    return network


def update_freeze_masks_resnet(network: Any) -> Any:
    network.freeze_masks = True
    return network


def update_freeze_masks(network: Any) -> Any:
    if isinstance(network, ResNet18):
        return update_freeze_masks_resnet(network)
    # Update freeze_masks
    weights = network.get_weight_bias_masks_numpy()
    freeze_masks = []
    mature_neurons = network.get_frozen_units()
    for i, target_mature in enumerate(mature_neurons[1:]):
        target_mature = np.array(target_mature, dtype=np.int32)
        mask_w = np.zeros(weights[i][0].shape)
        mask_b = np.zeros(weights[i][1].shape)
        if len(target_mature) != 0:
            mask_w[target_mature, :] = 1
            mask_b[target_mature] = 1
        freeze_masks.append((mask_w * weights[i][0], mask_b))

    freeze_masks = [(torch.tensor(w).to(torch.bool).to(get_device()),
                     torch.tensor(b).to(torch.bool).to(get_device()))
                    for w, b in freeze_masks]
    network.freeze_masks = freeze_masks

    for module in network.modules():
        if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm1d):
            layer_index = module.layer_index
            frozen_units = network.unit_ranks[layer_index][0] > 1
            module.freeze_units(torch.tensor(frozen_units, dtype=torch.bool).to(get_device()))  # type: ignore

    return network


def pick_top_neurons(scores, selection_ratio) -> List[int]:
    total = sum(scores)
    accumulate = 0
    indices = []
    sort_indices = torch.argsort(-scores)
    for index in sort_indices:
        if scores[index] == 0:
            continue
        index = int(index)
        accumulate = accumulate + scores[index]
        indices.append(index)
        if accumulate >= total * selection_ratio / 100.0:
            break
    return indices


def select_learner_units_resnet(network: Any, stable_selection_perc, train_episode: Any) -> Any:
    top_unit_indices = []
    index = 1
    if stable_selection_perc == 100.0:
        rank_dict = network.get_units_ranks_dict()
        for _, layer_name in network.layers[1:]:
            top_unit_indices.append((rank_dict[layer_name] < 2).nonzero()[0])
    else:
        bs = network.args.batch_size
        count = 0
        loader = DataLoader(train_episode.dataset, batch_size=bs,  shuffle=True)
        layer_activations = None
        with torch.no_grad():
            network.eval()
            for data, _, _ in loader:
                data = data.to(get_device())
                activations = network.get_activation_selection(data)
                count = count + bs
                if layer_activations is None:
                    layer_activations = activations
                else:
                    layer_activations = [torch.vstack((layer_activations[index], activation))
                                         for index, activation in enumerate(activations)]
                if count == 512:
                    break
        
        for layer_index, activations in enumerate(layer_activations[1:], 1): # type: ignore
            scores = torch.mean(activations, dim=(0, 2, 3))
            # Filter mature
            mask = torch.ones(scores.shape, dtype=torch.bool)
            mask[network.current_learner_neurons[layer_index]] = False
            scores[mask] = 0.0
            top_unit_indices.append(pick_top_neurons(scores, stable_selection_perc))

    # Add Episode Output Units
    top_unit_indices.append(train_episode.classes_in_this_experience)

    # Update current_young_neurons, current_learner_neurons, unit_ranks
    unit_ranks = [network.unit_ranks[0]] + []
    for index, layer_selected_units in enumerate(top_unit_indices, 1):
        new_ranks = copy.deepcopy(network.unit_ranks[index][0])
        new_ranks[new_ranks < 2] = 0
        new_ranks[layer_selected_units] = 1
        unit_ranks.append((new_ranks, network.unit_ranks[index][1]))

    network.unit_ranks = unit_ranks
    network.current_young_neurons = get_current_young_neurons(unit_ranks)
    network.current_learner_neurons = get_current_learner_neurons(unit_ranks)

    return network


def select_learner_units(network: Any, stable_selection_perc, train_episode: Any, episode_index) -> Any:
    if isinstance(network, ResNet18):
        return select_learner_units_resnet(network, stable_selection_perc, train_episode)
    
    top_unit_indices = []
    index = 1
    if stable_selection_perc == 100.0:
        for _, layer in network.named_children():
            if isinstance(layer, torch.nn.ModuleList):
                for _, sublayer in enumerate(layer):
                    if isinstance(sublayer, SparseConv2d) or isinstance(sublayer, SparseLinear):
                        top_unit_indices.append(network.current_learner_neurons[index] + network.current_young_neurons[index])
                        index = index + 1
            elif isinstance(layer, SparseLinear):
                top_unit_indices.append(network.current_learner_neurons[index] + network.current_young_neurons[index])
                index = index + 1
    else:
        loader = DataLoader(train_episode.dataset, batch_size=1024,  shuffle=False)
        data, _, _ = next(iter(loader))
        data = data.to(get_device())
        layer_activations = network.get_activation_selection(data)
        for _, layer in network.named_children():
            if isinstance(layer, torch.nn.ModuleList):
                for _, sublayer in enumerate(layer):
                    if isinstance(sublayer, SparseConv2d):
                        scores = torch.sum(layer_activations[index], axis=(0, 2, 3))  # type: ignore
                        # Filter mature
                        mask = torch.ones(scores.shape, dtype=torch.bool)
                        mask[network.current_learner_neurons[index]] = False
                        scores[mask] = 0.0
                        top_unit_indices.append(pick_top_neurons(scores, stable_selection_perc))
                        index = index + 1
                    elif isinstance(sublayer, SparseLinear):
                        scores = torch.sum(layer_activations[index], axis=0)  # type: ignore
                        # Filter mature
                        mask = torch.ones(scores.shape, dtype=torch.bool)
                        mask[network.current_learner_neurons[index]] = False
                        scores[mask] = 0.0
                        top_unit_indices.append(pick_top_neurons(scores, stable_selection_perc))
                        index = index + 1
            elif isinstance(layer, SparseLinear):
                scores = torch.sum(layer_activations[index], axis=0)  # type: ignore
                # Filter mature
                mask = torch.ones(scores.shape, dtype=torch.bool)
                mask[network.current_learner_neurons[index]] = False
                scores[mask] = 0.0
                top_unit_indices.append(pick_top_neurons(scores, stable_selection_perc))
                index = index + 1

    # Add Episode Output Units
    top_unit_indices.append(train_episode.classes_in_this_experience)
    # Update current_young_neurons, current_learner_neurons, unit_ranks
    unit_ranks = [network.unit_ranks[0]] + []
    for index, layer_selected_units in enumerate(top_unit_indices, 1):
        new_ranks = copy.deepcopy(network.unit_ranks[index][0])
        new_ranks[new_ranks < 2] = 0
        new_ranks[layer_selected_units] = 1
        unit_ranks.append((new_ranks, network.unit_ranks[index][1]))

    network.unit_ranks = unit_ranks
    network.current_young_neurons = get_current_young_neurons(unit_ranks)
    network.current_learner_neurons = get_current_learner_neurons(unit_ranks)
    return network


def grow_all_to_young_resnet(network: Any) -> Any:
    ranks_dict = network.get_units_ranks_dict()
    for module in network.modules():
        if isinstance(module, SparseConv2d):
            layer_name = module.layer_name
            if "->" in layer_name:
                prev_layer_name, layer_name = layer_name.split("->")
            else:
                prev_layer_name = "_".join(layer_name.split( "_")[:-1] + [str(int(layer_name.split("_")[-1])-1)])
            out_going_all_indices = (ranks_dict[prev_layer_name] > -1).nonzero()[0]
            in_coming_young_indices = (ranks_dict[layer_name] == 0).nonzero()[0]
            weight_mask, bias_mask = module.get_mask()
            weight_mask_np = weight_mask.cpu().numpy()
            weight_mask_np[np.ix_(in_coming_young_indices, out_going_all_indices)] = 1
            module.set_mask(torch.tensor(weight_mask_np, dtype=torch.bool), bias_mask)  # type: ignore

    return network


def grow_all_to_young(network: Any) -> Any:
    if isinstance(network, ResNet18):
        return grow_all_to_young_resnet(network)
    grow_masks = []
    connectivity_masks = [w for w, _ in network.get_weight_bias_masks_numpy()]
    all_young_indices = [list((u == 0).nonzero()[0])
                         for u, _ in network.unit_ranks]
    for i, next_layer_young_idx in enumerate(zip(all_young_indices[1:])):
        grow_mask = np.zeros(connectivity_masks[i].shape, dtype=np.intc)
        if next_layer_young_idx:
            grow_mask[next_layer_young_idx, :] = 1
        grow_masks.append(grow_mask)

    # Update Connections
    mask_index = 0
    for module in network.modules():
        if isinstance(module, SparseLinear) or isinstance(module, SparseConv2d):
            weight_mask, bias_mask = module.get_mask()
            weight_mask[torch.tensor(grow_masks[mask_index], dtype=torch.bool)] = 1
            module.set_mask(weight_mask, bias_mask)  # type: ignore
            mask_index = mask_index + 1
    return network


def drop_young_to_learner_resnet(network: Any) -> Any:
    ranks_dict = network.get_units_ranks_dict()
    for module in network.modules():
        if isinstance(module, SparseConv2d):
            layer_name = module.layer_name
            if "->" in layer_name:
                prev_layer_name, layer_name = layer_name.split("->")
            else:
                prev_layer_name = "_".join(layer_name.split("_")[:-1] + [str(int(layer_name.split("_")[-1])-1)])
            out_going_young_indices = (ranks_dict[prev_layer_name] == 0).nonzero()[0]
            in_coming_non_young_indices = (ranks_dict[layer_name] != 0).nonzero()[0]
            weight_mask, bias_mask = module.get_mask()
            weight_mask_np = weight_mask.cpu().numpy()
            weight_mask_np[np.ix_(in_coming_non_young_indices, out_going_young_indices)] = 0
            module.set_mask(torch.tensor(weight_mask_np, dtype=torch.bool), bias_mask)  # type: ignore
        elif isinstance(module, SparseOutput):
            out_going_young_indices = (ranks_dict[layer_name] == 0).nonzero()[0]  # type: ignore
            in_coming_non_young_indices = (ranks_dict["output"] != 0).nonzero()[0]
            weight_mask, bias_mask = module.get_mask()
            weight_mask_np = weight_mask.cpu().numpy()
            weight_mask_np[np.ix_(in_coming_non_young_indices, out_going_young_indices)] = 0
            module.set_mask(torch.tensor(weight_mask_np, dtype=torch.bool), bias_mask)  # type: ignore
    return network


def drop_young_to_learner(network: Any) -> Any:
    if isinstance(network, ResNet18):
        return drop_young_to_learner_resnet(network)
    # Create Drop masks
    drop_masks = []
    connectivity_masks = [w for w, _ in network.get_weight_bias_masks_numpy()]
    all_young_indices = [list((u == 0).nonzero()[0])
                         for u, _ in network.unit_ranks]
    all_not_young_indices = [list((u != 0).nonzero()[0])
                             for u, _ in network.unit_ranks]

    for i, (current_layer_young_idx, next_layer_not_young_idx) in enumerate(zip(
                                                all_young_indices[:-1], all_not_young_indices[1:])):
        drop_mask = np.zeros(connectivity_masks[i].shape, dtype=np.intc)
        if current_layer_young_idx:
            if len(connectivity_masks[i].shape) == 2 and len(connectivity_masks[i-1].shape) == 4:
                for young_index in current_layer_young_idx:
                    start = young_index*network.conv2lin_mapping_size
                    end = (young_index+1)*network.conv2lin_mapping_size
                    drop_mask[next_layer_not_young_idx, start:end] = 1
            else:
                drop_mask[np.ix_(next_layer_not_young_idx, current_layer_young_idx)] = 1
        drop_masks.append(drop_mask * connectivity_masks[i])

    # Update Connections
    mask_index = 0
    for module in network.modules():
        if isinstance(module, SparseLinear) or isinstance(module, SparseConv2d) or isinstance(module, SparseOutput):
            weight_mask, bias_mask = module.get_mask()
            weight_mask[torch.tensor(drop_masks[mask_index], dtype=torch.bool)] = 0
            module.set_mask(weight_mask, bias_mask)  # type: ignore
            mask_index = mask_index + 1
    return network
