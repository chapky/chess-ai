from pathlib import Path

import click
import wandb

from chess_ai.data.dataset import ChessDataset
from chess_ai.models.cnn.model import ChessAISmaller
from chess_ai.models.transformer.model import ChessTransformer
from chess_ai.training.trainer import train_model


@click.command()
@click.option(
    "--model-type",
    type=click.Choice(["cnn", "transformer"]),
    default="cnn",
    help="Model architecture to use",
)
@click.option(
    "--data-path",
    type=click.Path(exists=True),
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
    type=click.Path(),
    default="./models",
    help="Directory to save model checkpoints",
)
def train(
    model_type, data_path, rating_range, batch_size, epochs, learning_rate, save_dir
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
    dataset = ChessDataset.from_files(data_path, rating_range)

    # Create model
    if model_type == "cnn":
        model = ChessAISmaller()
    else:
        model = ChessTransformer()

    # Train
    train_model(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        save_dir=Path(save_dir),
    )


if __name__ == "__main__":
    train()
