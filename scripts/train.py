from pathlib import Path

import click
import torch
import wandb

from chess_ai.data.dataset import ChessPolicyDataset, ChessValueDataset
from chess_ai.models.cnn.model import ChessAISmaller, ChessAIValue
from chess_ai.models.transformer.model import ChessTransformer
from chess_ai.training.trainer import WandbCallback, train_policy_model, train_value_model


@click.command()
@click.option(
    "--model-arch",
    type=click.Choice(["cnn", "transformer"]),
    default="cnn",
    help="Model architecture to use",
)
@click.option(
    "--model-type",
    type=click.Choice(["policy", "value"]),
    default="policy",
    help="Type and purpose of the model to train",
)
@click.option(
    "--data-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to training data",
)
@click.option(
    "--rating-range",
    type=str,
    required=True,
    help='Rating range to train on (e.g. "1600-2000")',
)
@click.option("--batch-size", type=int, default=64, help="Training batch size")
@click.option("--epochs", type=int, default=10, help="Number of epochs to train")
@click.option("--learning-rate", type=float, default=0.001, help="Learning rate")
@click.option(
    "--save-dir",
    type=click.Path(path_type=Path),
    default="./models",
    help="Directory to save model checkpoints",
)
@click.option(
    "--from-checkpoint",
    type=click.Path(exists=True, path_type=Path),
    help="Path to a model checkpoint file (.pth) to resume training from",
    required=False,
)
@click.option(
    "--value-from-policy-path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to a policy model checkpoint file (.pth), from which to initialize thevalue model",
    required=False,
)
def train(
    model_arch: str,
    model_type: str,
    data_path: Path,
    rating_range: str,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    save_dir: Path,
    value_from_policy_path: Path | None = None,
    from_checkpoint: Path | None = None,
):
    """Train a chess model on Lichess game data."""

    # Initialize wandb
    wandb.init(
        project="chess_ai",
        config={
            "model_type": model_type,
            "rating_range": rating_range,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        },
    )

    # Load data
    if model_type == "value":
        dataset = ChessValueDataset.from_pgn(data_path, int(rating_range.split("-")[0]))
        if value_from_policy_path is not None:
            if from_checkpoint is not None:
                raise ValueError(
                    "Cannot specify both from_checkpoint and value_from_policy_path"
                )
            model = ChessAIValue.initialize_from_smaller_model(
                str(value_from_policy_path)
            )
        elif model_arch == "cnn":
            model = ChessAIValue()
        else:
            print("Value model ddoes not support other architectures than cnn yet")
    else:
        dataset = ChessPolicyDataset.from_files(data_path, rating_range)
        if model_arch == "cnn":
            model = ChessAISmaller()
        else:
            model = ChessTransformer()
    if from_checkpoint is not None:
        checkpoint = torch.load(from_checkpoint)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    # Train
    callback = WandbCallback()
    if model_type == "value":
        train_value_model(
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            save_dir=Path(save_dir),
            callback=callback,
        )
    else:
        train_policy_model(
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            save_dir=Path(save_dir),
            callback=callback,
        )


if __name__ == "__main__":
    train()
