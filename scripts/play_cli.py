from pathlib import Path

import chess
import click
import torch

from chess_ai.data.preprocessing import StandardEncoder
from chess_ai.models.cnn.model import ChessAISmaller
from chess_ai.models.transformer.model import ChessTransformer
from chess_ai.ui.base import GameController
from chess_ai.ui.cli import CliUI


def load_model(
    checkpoint_path: Path, model_type: str, device: torch.device
) -> ChessAISmaller | ChessTransformer:
    if model_type == "cnn":
        model = ChessAISmaller()
    else:
        model = ChessTransformer()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model


@click.command()
@click.option(
    "--checkpoint-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to model checkpoint file (.pth)",
)
@click.option(
    "--player-color",
    type=click.Choice(["white", "black"]),
    default="white",
    help="Color for the human player (white/black)",
)
@click.option(
    "--model-type",
    type=click.Choice(["cnn", "transformer"]),
    default="cnn",
    help="Model architecture to use",
)
def main(checkpoint_path: Path, player_color: str, model_type: str):
    """Play chess against a trained model via command line interface."""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = load_model(checkpoint_path, model_type, device)
    model = model.to(device)
    print("Model loaded successfully!")

    # Create UI and controller
    ui = CliUI()
    controller = GameController(
        ui=ui, model=model, encoder=StandardEncoder(), device=device
    )

    # Convert player color string to chess.Color
    color = chess.WHITE if player_color.lower() == "white" else chess.BLACK

    # Start game
    print("\nStarting new game...")
    print("Enter moves in UCI format (e.g. e2e4)")
    print("Enter 'q' to quit\n")
    controller.play_game(player_color=color)


if __name__ == "__main__":
    main()
