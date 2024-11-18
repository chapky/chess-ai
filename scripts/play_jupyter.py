from pathlib import Path

import chess
import click
import torch

from chess_ai.data.preprocessing import StandardEncoder
from chess_ai.models.cnn.model import ChessAISmaller
from chess_ai.ui.base import GameController
from chess_ai.ui.jupyter import JupyterUI


def load_model(checkpoint_path: str, device: torch.device) -> ChessAISmaller:
    model = ChessAISmaller()
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
def main(checkpoint_path: Path, player_color: str):
    """Play chess against a trained model via Jupyter notebook interface."""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model from checkpoint
    model = load_model(checkpoint_path, device)
    model = model.to(device)

    # Create UI and controller
    ui = JupyterUI()
    controller = GameController(
        ui=ui, model=model, encoder=StandardEncoder(), device=device
    )

    # Convert player color string to chess.Color
    color = chess.WHITE if player_color.lower() == "white" else chess.BLACK

    # Start a game
    controller.play_game(player_color=color)


if __name__ == "__main__":
    main()
