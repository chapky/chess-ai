from pathlib import Path
from typing import cast

import chess
import click
import torch

from chess_ai.data.preprocessing import StandardEncoder
from chess_ai.models.base import ChessPolicyModel, ChessValueModel
from chess_ai.models.cnn.model import ChessAISmaller, ChessAIValue
from chess_ai.models.mcts.model import MCTS
from chess_ai.models.transformer.model import (
    TransformerPolicyModel,
    TransformerValueModel,
)
from chess_ai.ui.base import GameController
from chess_ai.ui.pygame_ui import PygameUI

# can't import chess
# import chess


def load_model(
    checkpoint_path: Path, model_type: str, device: torch.device
) -> ChessPolicyModel | ChessValueModel:
    if model_type == "cnn":
        model = ChessAISmaller()
    elif model_type == "transformer":
        model = TransformerPolicyModel()
    elif model_type == "value-cnn":
        model = ChessAIValue()
    elif model_type == "value-transformer":
        model = TransformerValueModel()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)

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
@click.option(
    "--value-checkpoint-path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to value model checkpoint file (.pth). Automatically enables MCTS.",
)
@click.option(
    "--value-model-type",
    type=click.Choice(["cnn", "transformer"]),
    default=None,
    help="Value model architecture to use; defaults to the same as the policy model.",
)
def main(
    checkpoint_path: Path,
    player_color: str,
    model_type: str,
    value_checkpoint_path: Path | None = None,
    value_model_type: str | None = None,
):
    """Play chess against a trained model using PyGame interface."""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model: ChessPolicyModel | MCTS

    # Load model
    print("Loading model...")
    if value_checkpoint_path:
        policy_model = cast(
            ChessPolicyModel, load_model(checkpoint_path, model_type, device)
        )
        value_model = cast(
            ChessValueModel,
            load_model(
                value_checkpoint_path,
                "value-" + (value_model_type or model_type),
                device,
            ),
        )
        model = MCTS(device, policy_model, value_model, policy_model)
    else:
        model = cast(ChessPolicyModel, load_model(checkpoint_path, model_type, device))
    print("Model loaded successfully!")

    # Create UI and controller
    ui = PygameUI()
    controller = GameController(
        ui=ui, model=model, encoder=StandardEncoder(), device=device
    )

    # Convert player color string to chess.Color
    color = chess.WHITE if player_color.lower() == "white" else chess.BLACK

    # Start game
    print("\nStarting new game...")
    controller.play_game(player_color=color)


if __name__ == "__main__":
    main()
