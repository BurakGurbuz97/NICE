import torch.nn as nn
from Source.helper import SparseConv2d, SparseLinear, BatchNorm1Custom
from Source.helper import BatchNorm2Custom, get_device, SparseOutput
from argparse import Namespace
from torch.autograd import Function
from typing import List, Any, Tuple
import copy
import torch
import numpy as np


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, prev_layer_count=0):
        super(BasicBlock, self).__init__()
        self.conv1 = SparseConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False,
                                  layer_name="conv_early_{}".format(prev_layer_count + 1))
        self.bn1 = BatchNorm2Custom(out_channels, layer_name="conv_early_{}".format(prev_layer_count + 1))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = SparseConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False,
                                  layer_name="conv_early_{}".format(prev_layer_count + 2))
        self.bn2 = BatchNorm2Custom(out_channels, layer_name="conv_early_{}".format(prev_layer_count + 2))

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                SparseConv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False,
                             layer_name="conv_early_{}->conv_early_{}".format(prev_layer_count + 0, prev_layer_count + 2)),
                BatchNorm2Custom(out_channels, layer_name="conv_early_{}".format(prev_layer_count + 2))
            )

    def forward(self, x):
        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.bn2(self.conv2(out1))
        out = out2 + self.shortcut(x)
        out = self.relu(out)
        return out, torch.relu(out2), torch.relu(out1)


class ResNet18(nn.Module):
    def __init__(self, args: Namespace, input_size: int, output_size: int):
        super(ResNet18, self).__init__()
        self.args = args
        self.c = args.resnet_multiplier
        self.conv2lin_size = 512
        self.conv2lin_mapping_size = 1
        self.penultimate_layer_size = int(512 * self.c)
        self.input_size = input_size
        self.output_size = output_size
        self.conv1 = SparseConv2d(input_size, int(64 * self.c), kernel_size=3,
                                  stride=1, padding=1, bias=False, layer_name="conv_early_1")
        self.bn1 = BatchNorm2Custom(int(64 * self.c), layer_name="conv_early_1")
        self.relu = nn.ReLU(inplace=True)
        self.layers = [(3, "conv_early_0"), (int(64 * self.c), "conv_early_1"), (int(64 * self.c), "conv_early_2"),
                       (int(64 * self.c), "conv_early_3"), (int(64 * self.c), "conv_early_4"),
                       (int(64 * self.c), "conv_early_5"), (int(128 * self.c), "conv_early_6"),
                       (int(128 * self.c), "conv_early_7"), (int(128 * self.c), "conv_early_8"),
                       (int(128 * self.c), "conv_early_9"), (int(256 * self.c), "conv_early_10"),
                       (int(256 * self.c), "conv_early_11"), (int(256 * self.c), "conv_early_12"),
                       (int(256 * self.c), "conv_early_13"), (int(512 * self.c), "conv_early_14"),
                       (int(512 * self.c), "conv_early_15"), (int(512 * self.c), "conv_early_16"),
                       (int(512 * self.c), "conv_early_17")]

        # Block1
        self.block1_1 = BasicBlock(int(64 * self.c), int(64 * self.c), 1, 1)
        self.block1_2 = BasicBlock(int(64 * self.c), int(64 * self.c), 1, 3)

        # Block2
        self.block2_1 = BasicBlock(int(64 * self.c), int(128 * self.c), 2, 5)
        self.block2_2 = BasicBlock(int(128 * self.c), int(128 * self.c), 2, 7)

        # Block3
        self.block3_1 = BasicBlock(int(128 * self.c), int(256 * self.c), 2, 9)
        self.block3_2 = BasicBlock(int(256 * self.c), int(256 * self.c), 2, 11)

        # Block4
        self.block4_1 = BasicBlock(int(256 * self.c), int(512 * self.c), 2, 13)
        self.block4_2 = BasicBlock(int(512 * self.c), int(512 * self.c), 2, 15)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_layer = SparseOutput(int(512 * self.c), output_size, layer_name="output")

        self.classes_seen_so_far = []
        self.current_young_neurons = [[]] + [list(range(num_units)) for num_units, _ in self.layers[1:]] + [list(range(output_size))]
        self.current_learner_neurons = [[] for _, _ in self.layers]
        self.freeze_masks = []
        self.unit_ranks = [(np.array([999]*self.input_size), "conv_early_0")]
        for num_units, layer_name in self.layers[1:]:
            self.unit_ranks.append((np.array([0]*num_units), layer_name))

        self.unit_ranks.append((np.array([0]*output_size), "output"))  # type: ignore

    def get_units_ranks_dict(self):
        return {name: ranks for ranks, name in self.unit_ranks}

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        x, _, _ = self.block1_1(x)
        x, _, _ = self.block1_2(x)
        x, _, _ = self.block2_1(x)
        x, _, _ = self.block2_2(x)
        x, _, _ = self.block3_1(x)
        x, _, _ = self.block3_2(x)
        x, _, _ = self.block4_1(x)
        x, _, _ = self.block4_2(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

    def forward_output(self, x):
        x = self.forward(x)
        x = self.output_layer(x)
        x = Let_Learner.apply(x, self.current_learner_neurons[-1])
        return x

    def freeze_bn_layers(self):
        rank_dict = self.get_units_ranks_dict()
        for m in self.modules():
            if isinstance(m, (BatchNorm1Custom, BatchNorm2Custom)):
                layer_name = m.layer_name
                m.freeze_units(torch.tensor((rank_dict[layer_name] > 1)))  # type: ignore

    def add_seen_classes(self, classes: List) -> None:
        new_classes = set(self.classes_seen_so_far)
        for cls in classes:
            new_classes.add(cls)
        self.classes_seen_so_far = list(new_classes)

    def compute_weight_sparsity(self):
        parameters = 0
        ones = 0
        for module in self.modules():
            if isinstance(module, SparseLinear) or isinstance(module, SparseConv2d):
                shape = module.weight.data.shape
                parameters += torch.prod(torch.tensor(shape))
                w_mask, _ = copy.deepcopy(module.get_mask())
                ones += torch.count_nonzero(w_mask)
        return float((parameters - ones) / parameters) * 100

    def compute_weight_sparsity_2(self):
        parameters = 0
        ones = 0
        for module in self.modules():
            if isinstance(module, SparseLinear) or isinstance(module, SparseConv2d) or isinstance(module, SparseOutput):
                shape = module.weight.data.shape
                parameters += torch.prod(torch.tensor(shape))
                w_mask, _ = copy.deepcopy(module.get_mask())
                ones += torch.count_nonzero(w_mask)
        return (parameters, ones, float((parameters - ones) / parameters) * 100)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (SparseLinear, SparseConv2d, SparseOutput, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def set_masks(self, weight_masks: List[torch.Tensor], bias_masks: List[torch.Tensor]) -> None:
        i = 0
        for m in self.modules():
            if isinstance(m, (SparseLinear, SparseConv2d, SparseOutput)):
                m.set_mask(weight_masks[i], bias_masks[i])
                i = i + 1

    def get_activation_selection(self, x: torch.Tensor) -> List:
        rank_dict = self.get_units_ranks_dict()
        activations = [x.detach().cpu()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        activations.append(Let_Learner.apply(x.detach().cpu(),
                                            (rank_dict[self.conv1.layer_name] == 1).nonzero()))

        x, x2, x1 = self.block1_1(x)
        activations.append(Let_Learner.apply(x1.detach().cpu(),
                                             (rank_dict[self.block1_1.conv1.layer_name] == 1).nonzero()))
        activations.append(Let_Learner.apply(x2.detach().cpu(),
                                             (rank_dict[self.block1_1.conv2.layer_name] == 1).nonzero()))

        x, x2, x1 = self.block1_2(x)
        activations.append(Let_Learner.apply(x1.detach().cpu(),
                                             (rank_dict[self.block1_2.conv1.layer_name] == 1).nonzero()))
        activations.append(Let_Learner.apply(x2.detach().cpu(),
                                             (rank_dict[self.block1_2.conv2.layer_name] == 1).nonzero()))

        x, x2, x1 = self.block2_1(x)
        activations.append(Let_Learner.apply(x1.detach().cpu(),
                                             (rank_dict[self.block2_1.conv1.layer_name] == 1).nonzero()))
        activations.append(Let_Learner.apply(x2.detach().cpu(),
                                             (rank_dict[self.block2_1.conv2.layer_name] == 1).nonzero()))

        x, x2, x1 = self.block2_2(x2)
        activations.append(Let_Learner.apply(x1.detach().cpu(),
                                             (rank_dict[self.block2_2.conv1.layer_name] == 1).nonzero()))
        activations.append(Let_Learner.apply(x2.detach().cpu(),
                                             (rank_dict[self.block2_2.conv2.layer_name] == 1).nonzero()))

        x, x2, x1 = self.block3_1(x)
        activations.append(Let_Learner.apply(x1.detach().cpu(),
                                             (rank_dict[self.block3_1.conv1.layer_name] == 1).nonzero()))
        activations.append(Let_Learner.apply(x2.detach().cpu(),
                                             (rank_dict[self.block3_1.conv2.layer_name] == 1).nonzero()))

        x, x2, x1 = self.block3_2(x)
        activations.append(Let_Learner.apply(x1.detach().cpu(),
                                             (rank_dict[self.block3_2.conv1.layer_name] == 1).nonzero()))
        activations.append(Let_Learner.apply(x2.detach().cpu(),
                                             (rank_dict[self.block3_2.conv2.layer_name] == 1).nonzero()))

        x, x2, x1 = self.block4_1(x)
        activations.append(Let_Learner.apply(x1.detach().cpu(),
                                             (rank_dict[self.block4_1.conv1.layer_name] == 1).nonzero()))
        activations.append(Let_Learner.apply(x2.detach().cpu(),
                                             (rank_dict[self.block4_1.conv2.layer_name] == 1).nonzero()))

        x, x2, x1 = self.block4_2(x)
        activations.append(Let_Learner.apply(x1.detach().cpu(),
                                             (rank_dict[self.block4_2.conv1.layer_name] == 1).nonzero()))
        activations.append(Let_Learner.apply(x2.detach().cpu(),
                                             (rank_dict[self.block4_2.conv2.layer_name] == 1).nonzero()))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return activations

    def l2_loss(self):
        reg_terms = []
        for module in self.modules():
            if isinstance(module, (SparseLinear, SparseConv2d)):
                reg_terms.append(
                    torch.sum((module.weight * module.weight_mask)**2))
                reg_terms.append(torch.sum(module.bias**2))  # type: ignore
            elif isinstance(module, (nn.Linear, nn.Conv2d)):
                reg_terms.append(torch.sum(module.weight ** 2))
        return torch.sum(torch.stack(reg_terms))

    def get_activations(self, x: torch.Tensor, return_output=False) -> List:
        activations = [x.detach().cpu()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        activations.append(x.detach().cpu())

        x, x2, x1 = self.block1_1(x)
        activations.append(x1.detach().cpu())
        activations.append(x2.detach().cpu())

        x, x2, x1 = self.block1_2(x)
        activations.append(x1.detach().cpu())
        activations.append(x2.detach().cpu())

        x, x2, x1 = self.block2_1(x)
        activations.append(x1.detach().cpu())
        activations.append(x2.detach().cpu())

        x, x2, x1 = self.block2_2(x)
        activations.append(x1.detach().cpu())
        activations.append(x2.detach().cpu())

        x, x2, x1 = self.block3_1(x)
        activations.append(x1.detach().cpu())
        activations.append(x2.detach().cpu())

        x, x2, x1 = self.block3_2(x)
        activations.append(x1.detach().cpu())
        activations.append(x2.detach().cpu())

        x, x2, x1 = self.block4_1(x)
        activations.append(x1.detach().cpu())
        activations.append(x2.detach().cpu())

        x, x2, x1 = self.block4_2(x)
        activations.append(x1.detach().cpu())
        activations.append(x2.detach().cpu())

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.output_layer(x)
        activations.append(copy.deepcopy(x.detach().cpu()))
        if return_output:
            return activations[-1], activations  # type: ignore
        return activations

    def reset_frozen_gradients(self):
        ranks_dict = self.get_units_ranks_dict()
        for module in self.modules():
            if isinstance(module, SparseConv2d):
                layer_name = module.layer_name
                if "->" in layer_name:
                    prev_layer_name, layer_name = layer_name.split("->")
                else:
                    prev_layer_name = "_".join(layer_name.split("_")[:-1] + [str(int(layer_name.split("_")[-1])-1)])
                out_going_mature_indices = (
                    ranks_dict[prev_layer_name] > 1).nonzero()[0]
                in_coming_mature_indices = (
                    ranks_dict[layer_name] > 1).nonzero()[0]
                if module.weight.requires_grad:
                    grad_numpy = module.weight.grad.cpu().numpy()  # type: ignore
                    grad_numpy[np.ix_(in_coming_mature_indices, out_going_mature_indices)] = 0.0
                    module.weight.grad = torch.tensor(grad_numpy).to(get_device())
                    if module.bias_flag:
                        module.bias.grad[in_coming_mature_indices] = 0 # type: ignore
            if isinstance(module, SparseOutput):
                out_going_mature_indices = (ranks_dict[layer_name] > 1).nonzero()[0]  # type: ignore
                in_coming_mature_indices = (ranks_dict["output"] > 1).nonzero()[0]
                grad_numpy = module.weight.grad.cpu().numpy()  # type: ignore
                grad_numpy[np.ix_(in_coming_mature_indices, out_going_mature_indices)] = 0.0
                module.weight.grad = torch.tensor(grad_numpy).to(get_device())
                module.bias.grad[in_coming_mature_indices] = 0  # type: ignore
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                if module.frozen_units is not None and module.affine:
                    module.weight.grad = torch.where(module.frozen_units, torch.zeros_like(module.weight.grad), module.weight.grad)  # type: ignore
                    module.bias.grad = torch.where(module.frozen_units, torch.zeros_like(module.bias.grad), module.bias.grad)  # type: ignore


class Let_Learner(Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, learner_units: List) -> torch.Tensor:
        learner_units = np.array(learner_units)  # type: ignore
        indices = torch.tensor(learner_units, dtype=torch.long).to(x.device)
        new_x = torch.zeros_like(x)
        new_x[torch.arange(x.shape[0]).unsqueeze(1), learner_units] = x[torch.arange(x.shape[0]).unsqueeze(1), learner_units]
        ctx.save_for_backward(indices)
        return new_x

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any, Any, Any]:
        indices, = ctx.saved_tensors
        new_grad_output = torch.zeros_like(grad_output)
        new_grad_output[torch.arange(grad_output.shape[0]).unsqueeze(1), indices] = grad_output[torch.arange(grad_output.shape[0]).unsqueeze(1), indices]
        return new_grad_output, None, None, None
