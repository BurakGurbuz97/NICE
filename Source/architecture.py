from argparse import Namespace
import torch
import torch.nn as nn
from torch.autograd import Function
from numpy import typing as np_type
import numpy as np
from typing import Tuple, List, Any
import copy
from Source.helper import get_device, SparseConv2d, SparseLinear, SparseOutput
from Source.resnet18 import ResNet18


def get_backbone(args: Namespace, input_size: int, output_size: int) -> nn.Module:
    if args.model == "CNN_Simple":
        return CNN_Simple(args, input_size, output_size)
    elif args.model == "CNN_MNIST":
        return CNN_MNIST(args, input_size, output_size)
    elif args.model == "ResNet18":
        return ResNet18(args, input_size, output_size)
    elif args.model == "VGG11_SLIM":
        return VGG11_SLIM(args, input_size, output_size)
    else:
        raise Exception("args.model = {} is not defined!".format(args.model))


class CNN_Simple(nn.Module):
    def __init__(self, args: Namespace, input_size: int, output_size: int) -> None:
        super(CNN_Simple, self).__init__()
        self.args = args
        self.conv_layers_early = nn.ModuleList()
        self.conv_layers_late = nn.ModuleList()
        self.hidden_linear = nn.ModuleList()
        self.input_size = 1 if input_size == 784 else input_size
        self.output_size = output_size
        self.conv2lin_size = 128 * 4*4
        self.conv2lin_mapping_size = 4*4
        self.penultimate_layer_size = 1024
        self.conv_layers_early.extend([
            # Layer-1
            SparseConv2d(self.input_size, 64, 3, stride=1,layer_name="conv_early"), nn.ReLU(),
            # Layer-2
            SparseConv2d(64, 64, 3, stride=1, layer_name="conv_early"), nn.ReLU(), nn.MaxPool2d(2),
            # Layer-3
            SparseConv2d(64, 64, 3, stride=1, layer_name="conv_early"), nn.ReLU(),
            # Layer-4
            SparseConv2d(64, 64, 3, stride=1, layer_name="conv_early"), nn.ReLU(), nn.MaxPool2d(2),
        ])
        self.conv_layers_late.extend([
            # Layer-5
            SparseConv2d(64, 128, 3, stride=1, layer_name="conv_late"), nn.ReLU(),
            # Layer-6
            SparseConv2d(128, 128, 3, stride=1, layer_name="conv_late"), nn.ReLU(),
            # Layer-7
            SparseConv2d(128, 128, 3, stride=1, layer_name="conv_late"), nn.ReLU(), nn.MaxPool2d(2),
        ])
        self.hidden_linear.extend([
            SparseLinear(self.conv2lin_size, 1024, layer_name="linear"), nn.ReLU()
        ])

        self.penultimate_layer = nn.ModuleList([SparseLinear(1024, 1024, layer_name="linear"), nn.ReLU()])
        self.output_layer = SparseOutput(1024, output_size, layer_name="output")

        self.current_young_neurons = [[]] + [list(range(param.shape[0])) for param in self.parameters() if len(param.shape) != 1]
        self.current_learner_neurons = [list(range(0)) for param in self.parameters() if len(param.shape) != 1] + [[]]
        self.freeze_masks = []
        unit_ranks_list = [([999]*self.input_size, "input")]
        for m in self.modules():
            if isinstance(m, (SparseLinear, SparseConv2d, SparseOutput)):
                unit_ranks_list.append(
                    ([0]*m.weight.data.shape[0], m.layer_name))  # type: ignore
                
        self.unit_ranks = [(np.array(unit_types), name)
                           for unit_types, name in unit_ranks_list]

    def set_masks(self, weight_masks: List[torch.Tensor], bias_masks: List[torch.Tensor]) -> None:
        i = 0
        for m in self.modules():
            if isinstance(m, (SparseLinear, SparseConv2d, SparseOutput)):
                m.set_mask(weight_masks[i], bias_masks[i])
                i = i + 1

    # TODO: only resnet supports bn
    def freeze_bn_layers(self):
        return None

    def reinit_young(self):
        i = 0
        for m in self.modules():
            if isinstance(m, SparseLinear) or isinstance(m, SparseConv2d) or isinstance(m, SparseOutput):
                m.weight.data[self.current_young_neurons[i+1], :] = nn.init.kaiming_normal_(m.weight.data[self.current_young_neurons[i+1], :],
                                                                                            mode='fan_out', nonlinearity='relu')
                m.bias.data[self.current_young_neurons[i+1]] = nn.init.constant_(m.bias.data[self.current_young_neurons[i+1]], 0.0)   # type: ignore 
                i += 1

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (SparseLinear, SparseConv2d, nn.Linear, SparseOutput)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def get_activation_selection(self, x: torch.Tensor) -> List:
        activations = [torch.relu(x).detach().cpu()]
        index = 1
        for layer in self.conv_layers_early:
            x = layer(x)
            if isinstance(layer, (torch.nn.ReLU)):
                x_filter = Let_Learner.apply(x, self.current_learner_neurons[index])
                activations.append(x_filter.detach().cpu())  # type: ignore
                index = index + 1

        for layer in self.conv_layers_late:
            x = layer(x)
            if isinstance(layer, (torch.nn.ReLU)):
                x_filter = Let_Learner.apply(x, self.current_learner_neurons[index])
                activations.append(x_filter.detach().cpu())  # type: ignore
                index = index + 1

        x = x.view(-1, self.conv2lin_size)
        for layer in self.hidden_linear:
            x = layer(x)
            if isinstance(layer, (torch.nn.ReLU)):
                x_filter = Let_Learner.apply(x, self.current_learner_neurons[index])
                activations.append(x_filter.detach().cpu())
                index = index + 1

        for module in self.penultimate_layer:
            x = module(x)

        x_filter = Let_Learner.apply(x, self.current_learner_neurons[index])
        activations.append(x_filter.detach().cpu())
        return activations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.conv_layers_early:
            x = layer(x)
        for layer in self.conv_layers_late:
            x = layer(x)
        x = x.view(-1, self.conv2lin_size)
        for layer in self.hidden_linear:
            x = layer(x)
        for module in self.penultimate_layer:
            x = module(x)
        x = MaskedOut_Young.apply(x, self.current_young_neurons[-1])
        return x

    def forward_output(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward(x)
        x = self.output_layer(x)
        x = Let_Learner.apply(x, self.current_learner_neurons[-1])
        return x

    def get_activations(self, x: torch.Tensor, return_output=False) -> List[torch.Tensor]:
        activations = [torch.relu(x).detach().cpu()]
        for layer in self.conv_layers_early:
            x = layer(x)
            if isinstance(layer, (torch.nn.ReLU)):
                activations.append(copy.deepcopy(x.detach().cpu()))

        for layer in self.conv_layers_late:
            x = layer(x)
            if isinstance(layer, (torch.nn.ReLU)):
                activations.append(copy.deepcopy(x.detach().cpu()))

        x = x.view(-1, self.conv2lin_size)
        for layer in self.hidden_linear:
            x = layer(x)
            if isinstance(layer, (torch.nn.ReLU)):
                activations.append(copy.deepcopy(x.detach().cpu()))

        for module in self.penultimate_layer:
            x = module(x)
        activations.append(copy.deepcopy(x.detach().cpu()))

        x = self.output_layer(x)
        activations.append(copy.deepcopy(x.detach().cpu()))
        if return_output:
            return activations[-1], activations  # type: ignore
        return activations

    def l2_loss(self):
        reg_terms = []
        for module in self.modules():
            if isinstance(module, (SparseLinear, SparseConv2d, SparseOutput)):
                reg_terms.append(torch.sum((module.weight * module.weight_mask)**2))
                reg_terms.append(torch.sum(module.bias**2))  # type: ignore
        return torch.sum(torch.stack(reg_terms))

    def get_weight_bias_masks_numpy(self) -> List[Tuple[np_type.NDArray[np.double], np_type.NDArray[np.double]]]:
        weights = []
        for module in self.modules():
            if isinstance(module, SparseLinear) or isinstance(module, SparseConv2d) or isinstance(module, SparseOutput):
                weight_mask, bias_mask = module.get_mask()  # type: ignore
                weights.append((copy.deepcopy(weight_mask).cpu().numpy(),
                                copy.deepcopy(bias_mask).cpu().numpy()))  # type: ignore
        return weights

    def compute_weight_sparsity(self):
        parameters = 0
        ones = 0
        for module in self.modules():
            if isinstance(module, SparseLinear) or isinstance(module, SparseConv2d) or isinstance(module, SparseOutput):
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

    def reset_frozen_gradients(self):
        mask_index = 0
        for module in self.modules():
            if isinstance(module, SparseLinear) or isinstance(module, SparseConv2d) or isinstance(module, SparseOutput):
                if module.weight.requires_grad:
                    module.weight.grad[self.freeze_masks[mask_index][0]] = 0 # type: ignore
                    if module.bias_flag:
                        module.bias.grad[self.freeze_masks[mask_index][1]] = 0 # type: ignore
                mask_index = mask_index + 1

            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                if module.frozen_units is not None and module.affine:
                    module.weight.grad = torch.where(module.frozen_units, torch.zeros_like(module.weight.grad), module.weight.grad)  # type: ignore
                    module.bias.grad = torch.where(module.frozen_units, torch.zeros_like(module.bias.grad), module.bias.grad)  # type: ignore

    def get_frozen_units(self) -> List:
        frozen_units = []
        for unit_layer, _ in self.unit_ranks:
            frozen_units.append((unit_layer > 1).nonzero()[0])
        return frozen_units


class VGG11_SLIM(CNN_Simple):

    def __init__(self, args: Namespace, input_size: int, output_size: int) -> None:
        super(CNN_Simple, self).__init__()
        self.args = args
        self.conv_layers_early = nn.ModuleList()
        self.conv_layers_late = nn.ModuleList()
        self.hidden_linear = nn.ModuleList()
        self.input_size = input_size
        self.output_size = output_size
        self.conv2lin_size = 256 * 2 * 2
        self.conv2lin_mapping_size = 2*2
        self.penultimate_layer_size = 1024
        self.conv_layers_early.append(nn.Identity())
        self.conv_layers_late.extend([
            SparseConv2d(3, 32, kernel_size=3, padding=1, layer_name="conv_late"),
            nn.ReLU(inplace=True),

            SparseConv2d(32, 64, kernel_size=3, padding=1, layer_name="conv_late"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            SparseConv2d(64, 128, kernel_size=3, padding=1,layer_name="conv_late"),
            nn.ReLU(inplace=True),
            SparseConv2d(128, 128, kernel_size=3, padding=1, layer_name="conv_late"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            SparseConv2d(128, 256, kernel_size=3, padding=1, layer_name="conv_late"),
            nn.ReLU(inplace=True),
            SparseConv2d(256, 256, kernel_size=3, padding=1, layer_name="conv_late"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            SparseConv2d(256, 256, kernel_size=3, padding=1, layer_name="conv_late"),
            nn.ReLU(inplace=True),
            SparseConv2d(256, 256, kernel_size=3, padding=1, layer_name="conv_late"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])
        self.hidden_linear.extend([
            SparseLinear(self.conv2lin_size, 1024, layer_name="linear"), nn.ReLU(),
        ])
        self.penultimate_layer = nn.ModuleList([SparseLinear(1024, 1024, layer_name="linear"), nn.ReLU()])

        self.output_layer = SparseOutput(1024, output_size, layer_name="output")
        if self.output_size == 100:
            self._initialize_weights()
        self.current_young_neurons = [[]] + [list(range(param.shape[0])) for param in self.parameters() if len(param.shape) != 1]
        self.current_learner_neurons = [ list(range(0)) for param in self.parameters() if len(param.shape) != 1] + [[]]

        self.freeze_masks = []
        unit_ranks_list = [([999]*self.input_size, "input")]
        for m in self.modules():
            if isinstance(m, (SparseLinear, SparseConv2d, SparseOutput)):
                unit_ranks_list.append(([0]*m.weight.data.shape[0], m.layer_name))  # type: ignore
        self.unit_ranks = [(np.array(unit_types), name)
                           for unit_types, name in unit_ranks_list]


class CNN_MNIST(CNN_Simple):
    def __init__(self, args: Namespace, input_size: int, output_size: int) -> None:
        super(CNN_MNIST, self).__init__(args, input_size, output_size)
        self.args = args
        self.conv_layers_early = nn.ModuleList()
        self.conv_layers_late = nn.ModuleList()
        self.hidden_linear = nn.ModuleList()
        self.input_size = 1 if input_size == 784 else input_size
        self.output_size = output_size
        self.conv2lin_size = 32 * 7 * 7
        self.conv2lin_mapping_size = 7 * 7
        self.penultimate_layer_size = 500
        self.conv_layers_early.extend([
            SparseConv2d(self.input_size, 32, 3, stride=1, padding=1,
                         layer_name="conv_early"), nn.ReLU(), nn.MaxPool2d(2),
            SparseConv2d(32, 32, 3, stride=1, padding=1,
                         layer_name="conv_early"), nn.ReLU(), nn.MaxPool2d(2),
        ])
        self.conv_layers_late.append(nn.Identity())
        self.hidden_linear.append(nn.Identity())

        self.penultimate_layer = nn.ModuleList([SparseLinear(32*7*7, 500, layer_name="linear"), nn.ReLU()])
        self.output_layer = SparseOutput(500, output_size, layer_name="output")
        
        self.current_young_neurons = [[]] + [list(range(param.shape[0])) for param in self.parameters(
        ) if len(param.shape) != 1] + [list(range(output_size))]

        self.current_learner_neurons = [list(range(0)) for param in self.parameters() if len(param.shape) != 1] + [[]]

        self.freeze_masks = []
        unit_ranks_list = [([999]*self.input_size, "input")]
        for m in self.modules():
            if isinstance(m, (SparseLinear, SparseConv2d, SparseOutput)):
                unit_ranks_list.append(([0]*m.weight.data.shape[0], m.layer_name))  # type: ignore
        self.unit_ranks = [(np.array(unit_types), name)
                           for unit_types, name in unit_ranks_list]


class MaskedOut_Young(Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, young_units: List) -> torch.Tensor:
        indices = torch.tensor(young_units, dtype=torch.long).to(get_device())
        new_x = x.clone()
        new_x[torch.arange(x.shape[0]).unsqueeze(1), young_units] = 0
        ctx.save_for_backward(indices)
        return new_x

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any, Any, Any]:
        indices, = ctx.saved_tensors
        grad_output[torch.arange(
            grad_output.shape[0]).unsqueeze(1), indices] = 0
        return grad_output, None, None, None


class Let_Learner(Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, learner_units: List) -> torch.Tensor:
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
