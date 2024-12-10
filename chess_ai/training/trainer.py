import logging
from pathlib import Path
from typing import Protocol

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch import Tensor
from tqdm import tqdm

from chess_ai.models.base import ChessPolicyModel, ChessValueModel

from chess_ai.data.dataset import ChessPolicyDataset, ChessValueDataset

logger = logging.getLogger(__name__)


class TrainingCallback(Protocol):
    """Protocol for training callbacks."""

    def on_epoch_start(self, epoch: int) -> None:
        """Called at the start of each epoch."""
        ...

    def on_batch_end(self, epoch: int, batch: int, loss: float, metrics: dict) -> None:
        """Called after each batch."""
        ...

    def on_epoch_end(self, epoch: int, avg_loss: float, metrics: dict) -> None:
        """Called at the end of each epoch."""
        ...


class WandbCallback:
    """Training callback that logs to Weights & Biases."""

    def on_epoch_start(self, epoch: int) -> None:
        wandb.log({"epoch": epoch})

    def on_batch_end(self, epoch: int, batch: int, loss: float, metrics: dict) -> None:
        wandb.log({"loss": loss, **metrics})

    def on_epoch_end(self, epoch: int, avg_loss: float, metrics: dict) -> None:
        wandb.log({"epoch_loss": avg_loss, **metrics})


def train_policy_model(
    model: ChessPolicyModel,
    dataset: ChessPolicyDataset,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    save_dir: Path,
    callback: TrainingCallback | None = None,
    device: torch.device | None = None,
) -> None:
    """Train a chess model.

    Args:
        model: The model to train
        dataset: Training dataset
        batch_size: Batch size
        epochs: Number of epochs
        learning_rate: Learning rate
        save_dir: Directory to save checkpoints
        callback: Optional training callback
        device: Device to train on (defaults to cuda if available)
    """
    # Setup
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss()

    dataloader = dataset.create_dataloader(batch_size)

    save_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        if callback:
            callback.on_epoch_start(epoch)

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, (states, consts, moves) in enumerate(progress_bar):
            states = states.to(device)
            consts = consts.to(device)
            moves = moves.to(device)

            optimizer.zero_grad()

            outputs = model(states, consts)
            loss = loss_fn(outputs, moves)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

            # Calculate metrics
            metrics = {
                "top_1_acc": calculate_top_k_accuracy(outputs, moves, k=1),
                "top_5_acc": calculate_top_k_accuracy(outputs, moves, k=5),
            }

            if callback:
                callback.on_batch_end(epoch, i, loss.item(), metrics)

            # Update progress bar
            progress_bar.set_postfix(
                {"loss": f"{loss.item():.3f}", "top1": f'{metrics["top_1_acc"]:.3f}'}
            )

        # End of epoch
        avg_loss = running_loss / len(dataloader)
        logger.info("Epoch %d - Avg loss: %.3f", epoch + 1, avg_loss)

        if callback:
            callback.on_epoch_end(epoch, avg_loss, metrics)

        # Save checkpoint
        checkpoint_path = save_dir / f"policy_model_epoch_{epoch+1}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_loss,
            },
            checkpoint_path,
        )

        scheduler.step()


def calculate_top_k_accuracy(output, target, k=5):
    """Calculate top-k accuracy."""
    with torch.no_grad():
        _, pred = output.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return correct[:k].float().sum(0, keepdim=True).mean().item()


def calculate_metrics(outputs: Tensor, targets: Tensor) -> dict[str, float]:
    """Calculate training metrics.

    Args:
        outputs: Model outputs
        targets: True targets

    Returns:
        Dictionary of metric names and values
    """
    with torch.no_grad():
        metrics = {}
        for k in [1, 3, 5, 10]:
            acc = calculate_top_k_accuracy(outputs, targets, k)
            metrics[f"top_{k}_acc"] = acc

        # Calculate move type accuracies
        probs = torch.softmax(outputs, dim=1)

        # Piece movement accuracy
        piece_mask = targets < 4800
        if piece_mask.any():
            piece_outputs = outputs[piece_mask, :4800]
            piece_targets = targets[piece_mask]
            piece_pred = torch.argmax(piece_outputs, dim=1)
            metrics["piece_move_acc"] = (
                (piece_pred == piece_targets).float().mean().item()
            )
        else:
            metrics["piece_move_acc"] = 0.0

        # Promotion accuracy
        prom_mask = targets >= 4800
        if prom_mask.any():
            prom_outputs = outputs[prom_mask, 4800:]
            prom_targets = targets[prom_mask] - 4800
            prom_pred = torch.argmax(prom_outputs, dim=1)
            metrics["promotion_acc"] = (prom_pred == prom_targets).float().mean().item()
        else:
            metrics["promotion_acc"] = 0.0

        return metrics


def train_value_model(
    model: ChessValueModel,
    dataset: ChessValueDataset,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    save_dir: Path,
    callback: TrainingCallback | None = None,
    device: torch.device | None = None,
) -> None:
    """Train a chess value model with only the last layer unfrozen.

    Args:
        model: The model to train
        dataset: Training dataset
        batch_size: Batch size
        epochs: Number of epochs
        learning_rate: Learning rate
        save_dir: Directory to save checkpoints
        callback: Optional training callback
        device: Device to train on (defaults to cuda if available)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    model.fc2.weight.requires_grad = True
    model.fc2.bias.requires_grad = True

    optimizer = optim.Adam(
        [{"params": model.fc2.parameters()}], lr=learning_rate, weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    loss_fn = nn.MSELoss()

    dataloader = dataset.create_dataloader(batch_size)

    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        if callback:
            callback.on_epoch_start(epoch)

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        epoch_metrics = {
            "loss": 0.0,
            "mae": 0.0,
        }

        for i, (states, consts, outcomes) in enumerate(progress_bar):
            states = states.to(device)
            consts = consts.to(device)
            outcomes = outcomes.to(device).float().unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(states, consts)
            loss = loss_fn(outputs, outcomes)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()

            with torch.no_grad():
                mae = torch.mean(torch.abs(outputs - outcomes)).item()

            running_loss += loss.item()

            epoch_metrics["loss"] += loss.item()
            epoch_metrics["mae"] += mae

            if callback:
                callback.on_batch_end(
                    epoch, i, loss.item(), {"batch_loss": loss.item(), "batch_mae": mae}
                )

            progress_bar.set_postfix(
                {"loss": f"{loss.item():.4f}", "mae": f"{mae:.4f}"}
            )

        for key in epoch_metrics:
            epoch_metrics[key] /= len(dataloader)

        logger.info(
            "Epoch %d - Avg Loss: %.4f, Avg MAE: %.4f",
            epoch + 1,
            epoch_metrics["loss"],
            epoch_metrics["mae"],
        )

        if callback:
            callback.on_epoch_end(epoch, epoch_metrics["loss"], epoch_metrics)

        checkpoint_path = save_dir / f"value_model_epoch_{epoch+1}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": epoch_metrics["loss"],
            },
            checkpoint_path,
        )

        scheduler.step()


def calculate_value_metrics(outputs: Tensor, targets: Tensor) -> dict[str, float]:
    """Calculate training metrics for value model.

    Args:
        outputs: Model outputs
        targets: True targets

    Returns:
        Dictionary of metric names and values
    """
    with torch.no_grad():
        mae = torch.mean(torch.abs(outputs - targets)).item()
        mse = torch.mean((outputs - targets) ** 2).item()

        # Binary classification metrics
        binary_preds = (outputs >= 0.5).float()
        binary_targets = (targets >= 0.5).float()

        accuracy = (binary_preds == binary_targets).float().mean().item()
        precision = (binary_preds * binary_targets).sum() / binary_preds.sum()
        recall = (binary_preds * binary_targets).sum() / binary_targets.sum()

        f1_score = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "mae": mae,
            "mse": mse,
            "accuracy": accuracy,
            "precision": precision.item(),
            "recall": recall.item(),
            "f1_score": f1_score,
        }
