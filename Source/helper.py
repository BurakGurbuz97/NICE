from argparse import Namespace
import torch
from torch.autograd import Variable
from typing import Tuple, Optional, Any
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from avalanche.benchmarks import TCLExperience
from torch.utils.data import DataLoader


def get_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def to_var(x, requires_grad=False, volatile=False) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        x = x.clone().detach().to(get_device())
    else:
        x = torch.tensor(x).to(get_device())
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


class SparseConv2d(nn.Conv2d):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1,
                 padding=1, dilation=1, groups=1, bias=True, layer_name=""):
        # We keep bias as redundant variable for compatibility
        super(SparseConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                           stride, padding, dilation, groups, bias=True)
        self.layer_name = layer_name
        self.bias_flag = bias

    def set_mask(self, weight_mask: torch.Tensor, bias_mask: torch.Tensor) -> None:
        self.weight_mask = to_var(weight_mask, requires_grad=False)
        self.weight.data = self.weight.data * self.weight_mask.data

        self.bias_mask = to_var(bias_mask, requires_grad=False)
        self.bias.data = self.bias.data * self.bias_mask.data  # type: ignore

    def get_mask(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.weight_mask, self.bias_mask

    def forward(self, x) -> torch.Tensor:
        weight = self.weight * self.weight_mask
        bias = self.bias * self.bias_mask
        return F.conv2d(x, weight, bias if self.bias_flag else None,
                        self.stride, self.padding, self.dilation, self.groups)


class SparseLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias_flag=True, layer_name=""):
        super(SparseLinear, self).__init__(in_features, out_features, True)
        self.bias_flag = bias_flag
        self.layer_name = layer_name

    def set_mask(self, weight_mask: torch.Tensor, bias_mask: torch.Tensor) -> None:
        self.weight_mask = to_var(weight_mask, requires_grad=False)
        self.weight.data = self.weight.data * self.weight_mask.data

        self.bias_mask = to_var(bias_mask, requires_grad=False)
        self.bias.data = self.bias.data * self.bias_mask.data

    def get_mask(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.weight_mask, self.bias_mask

    def forward(self, x) -> torch.Tensor:
        weight = self.weight * self.weight_mask
        bias = self.bias * self.bias_mask
        return F.linear(x, weight, bias if self.bias_flag else None)


class SparseOutput(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias_flag=True, layer_name=""):
        super(SparseOutput, self).__init__(in_features, out_features, True)
        self.bias_flag = bias_flag
        self.layer_name = layer_name

    def set_mask(self, weight_mask: torch.Tensor, bias_mask: torch.Tensor) -> None:
        self.weight_mask = to_var(weight_mask, requires_grad=False)
        self.weight.data = self.weight.data * self.weight_mask.data

        self.bias_mask = to_var(bias_mask, requires_grad=False)
        self.bias.data = self.bias.data * self.bias_mask.data

    def get_mask(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.weight_mask, self.bias_mask

    def forward(self, x) -> torch.Tensor:
        weight = self.weight * self.weight_mask
        bias = self.bias * self.bias_mask
        return F.linear(x, weight, bias if self.bias_flag else None)


def reduce_or_flat_convs(activations, reduce=True):
    processed_activations = []
    is_conv = []
    for activation in activations:
        if len(activation.shape) == 4:
            if reduce:
                if activation.shape[2:] == (1, 1):
                    processed_activations.append(torch.squeeze(activation))
                else:
                    processed_activations.append(torch.mean(activation, dim=(2, 3)))
            else:
                processed_activations.append(activation.view(activation.shape[0], -1))
            is_conv.append(True)
        else:
            processed_activations.append(activation)
            is_conv.append(False)
    return is_conv, processed_activations


def random_prune(network: Any, pruning_perc: float, skip_first_conv=True) -> nn.Module:
    pruning_perc = pruning_perc / 100.0
    weight_masks, bias_masks = [], []
    first_conv_flag = skip_first_conv
    for module in network.modules():
        if isinstance(module, (SparseLinear, SparseOutput)):
            weight_masks.append(torch.from_numpy(np.random.choice([0, 1], module.weight.shape,
                                                 p=[pruning_perc, 1 - pruning_perc])))
            bias_masks.append(torch.ones(module.bias.shape))
        if isinstance(module, SparseConv2d):
            connectivity_mask = torch.from_numpy(np.random.choice([0, 1],
                                                                  (module.weight.shape[0],
                                                                   module.weight.shape[1]),
                                                                  p=[0, 1] if first_conv_flag else [pruning_perc, 1 - pruning_perc]))
            first_conv_flag = False
            in_range, out_range = range(module.weight.shape[1]), range(module.weight.shape[0])
            kernel_shape = (module.weight.shape[2], module.weight.shape[3])
            filter_masks = [[np.ones(kernel_shape) if connectivity_mask[out_index, in_index] else np.zeros(kernel_shape)
                            for in_index in in_range] for out_index in out_range]
            weight_masks.append(torch.from_numpy(np.array(filter_masks)).to(torch.float32))
            bias_masks.append(torch.ones(module.bias.shape).to(torch.float32))  # type: ignore
    network.set_masks(weight_masks, bias_masks)
    network.to(get_device())
    return network


def get_data_loaders(args: Namespace,
                     train_task: TCLExperience, val_task: TCLExperience,
                     test_task: TCLExperience, task_index: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:

    return _get_data_loaders_gpu(args, train_task, val_task, test_task, task_index)


def _get_data_loaders_gpu(args: Namespace, train_task: TCLExperience,
                          val_task: TCLExperience, test_task: TCLExperience, task_index: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    loaders = []
    for task_dataset, skip_last, bs in [(train_task.dataset, True, args.batch_size),
                                        (val_task.dataset, True, args.batch_size),
                                        (test_task.dataset, False, args.batch_size)]:
        x, y, t = zip(*[(x, y, t) for x, y, t in task_dataset])
        x_tensor = torch.stack(x).clone().detach().to("cuda")
        y_tensor, t_tensor = torch.tensor(y).to("cuda"), torch.tensor(t).to("cuda")
        tensor_dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor, t_tensor)  # type: ignore
        loaders.append(DataLoader(tensor_dataset, batch_size=bs,
                       shuffle=True, drop_last=skip_last))
    return tuple(loaders)  # type: ignore


class BatchNorm1Custom(torch.nn.BatchNorm1d):
    def __init__(self, num_features, layer_index, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm1Custom, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.frozen_units = None
        self.layer_index = layer_index
        self.running_mean_frozen = None
        self.running_var_frozen = None

    def freeze_units(self, frozen_units):
        if self.frozen_units is None:
            self.frozen_units = frozen_units
            self.running_mean_frozen = self.running_mean.data.clone()  # type: ignore
            self.running_var_frozen = self.running_var.data.clone()  # type: ignore
        else:
            new_frozen_units = torch.logical_xor(self.frozen_units, frozen_units)
            self.running_mean_frozen = torch.where(new_frozen_units, self.running_mean.data, self.running_mean_frozen)  # type: ignore
            self.running_var_frozen = torch.where(new_frozen_units, self.running_var.data, self.running_var_frozen)  # type: ignore
            self.frozen_units = frozen_units

    def forward(self, input):
        if self.frozen_units is not None:
            # Replace the frozen dimensions in self.running_mean and running_var
            self.running_mean.data = torch.where(self.frozen_units, self.running_mean_frozen, self.running_mean.data)  # type: ignore
            self.running_var.data = torch.where(self.frozen_units, self.running_var_frozen, self.running_var.data)  # type: ignore

        # Call the forward method of the parent class
        return super(BatchNorm1Custom, self).forward(input)


class BatchNorm2Custom(torch.nn.BatchNorm2d):
    def __init__(self, num_features, layer_name, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm2Custom, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.frozen_units = None
        self.layer_name = layer_name
        self.running_mean_frozen = None
        self.running_var_frozen = None

    def freeze_units(self, frozen_units):
        frozen_units = frozen_units.to(get_device())
        if self.frozen_units is None:
            self.frozen_units = frozen_units
            self.running_mean_frozen = self.running_mean.data.clone()  # type: ignore
            self.running_var_frozen = self.running_var.data.clone()  # type: ignore
        else:
            new_frozen_units = torch.logical_xor(self.frozen_units, frozen_units)
            self.running_mean_frozen = torch.where(
                new_frozen_units, self.running_mean.data, self.running_mean_frozen)   # type: ignore
            self.running_var_frozen = torch.where(
                new_frozen_units, self.running_var.data, self.running_var_frozen)  # type: ignore
            self.frozen_units = frozen_units

    def forward(self, input):
        if self.frozen_units is not None:
            # Replace the frozen dimensions in self.running_mean and running_var
            self.running_mean.data = torch.where(self.frozen_units, self.running_mean_frozen, self.running_mean.data)  # type: ignore
            self.running_var.data = torch.where(self.frozen_units, self.running_var_frozen, self.running_var.data)  # type: ignore

        # Call the forward method of the parent class
        return super(BatchNorm2Custom, self).forward(input)
