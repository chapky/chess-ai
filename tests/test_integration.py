import chess
import torch

from chess_ai.data.preprocessing import StandardEncoder
from chess_ai.models.cnn.model import ChessAISmaller
from chess_ai.ui.base import GameController
from chess_ai.ui.jupyter import JupyterUI


def test_game_flow():
    # Initialize components
    model = ChessAISmaller()
    ui = JupyterUI()
    encoder = StandardEncoder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create game controller
    controller = GameController(ui=ui, model=model, encoder=encoder, device=device)

    # Test game play
    try:
        controller.play_game(player_color=chess.WHITE)
        print("Game flow test passed!")
    except Exception as e:
        print(f"Game flow test failed: {str(e)}")


if __name__ == "__main__":
    test_game_flow()
