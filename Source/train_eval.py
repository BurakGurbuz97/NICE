from argparse import Namespace
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Any
import numpy as np

from Source.helper import get_device


def test(network: Any, context_detector: Any, data_loader: DataLoader, episode_id=None, return_preds=False) -> float:
    network.eval()
    predictions = []
    ground_truths = []
    episode_preds_all = []
    with torch.no_grad():
        for data, target, _ in data_loader:
            data = data.to(get_device())
            target = target.to(get_device())
            output, activations = network.get_activations(data, return_output=True)
            class_preds, episode_preds = context_detector.predict_context(activations, episode_id)
            for index, episode_pred in enumerate(class_preds):
                output[index, episode_pred] = output[index, episode_pred] + 99999
            preds = output.argmax(dim=1, keepdim=True)
            predictions.extend(preds)
            ground_truths.extend(target)
            episode_preds_all.extend(episode_preds)

    predictions = np.array([int(p) for p in predictions])
    ground_truths = np.array([int(gt) for gt in ground_truths])
    network.train()
    if return_preds:
        return sum(predictions == ground_truths) / len(predictions), predictions, ground_truths, episode_preds_all # type: ignore
    else:
        return sum(predictions == ground_truths) / len(predictions)


def phase_training_ce(network: Any, phase_epochs: int,
                      loss: nn.Module, optimizer: ..., train_loader: DataLoader, args: Namespace) -> Any:
    for _ in range(phase_epochs):
        network.train()
        epoch_l2_loss = []
        epoch_ce_loss = []
        for data, target, _ in train_loader:
            data = data.to(get_device())
            target = target.to(get_device())
            optimizer.zero_grad()
            stream_output = network.forward_output(data)
            ce_loss = loss(stream_output, target.long())
            reg_loss = (args.weight_decay * network.l2_loss())
            epoch_ce_loss.append(ce_loss)
            epoch_l2_loss.append(reg_loss)
            batch_loss = reg_loss + ce_loss
            batch_loss.backward()
            if network.freeze_masks:
                network.reset_frozen_gradients()
            optimizer.step()
        print("Average training loss input: {}".format(
            sum(epoch_ce_loss) / len(epoch_ce_loss)))
        print("Average l2 loss: {}".format(
            sum(epoch_l2_loss) / len(epoch_l2_loss)))

    return network
