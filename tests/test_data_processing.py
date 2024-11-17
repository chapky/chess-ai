import torch

from chess_ai.utils.chess_utils import tensor_to_board


def test_tensor_to_board():
    # Create a sample tensor representing initial chess position
    tensor = torch.zeros(8, 8, 12)

    # This is an impossible board because copilot go brr
    # Set up initial position pieces
    # White pieces (first 6 channels)
    tensor[0, 0:8, 3] = 1  # Rooks
    tensor[0, 1:7:5, 1] = 1  # Knights
    tensor[0, 2:6:3, 2] = 1  # Bishops
    tensor[0, 3, 4] = 1  # Queen
    tensor[0, 4, 5] = 1  # King
    tensor[1, 0:8, 0] = 1  # Pawns

    # Black pieces (last 6 channels)
    tensor[7, 0:8, 9] = 1  # Rooks
    tensor[7, 1:7:5, 7] = 1  # Knights
    tensor[7, 2:6:3, 8] = 1  # Bishops
    tensor[7, 3, 10] = 1  # Queen
    tensor[7, 4, 11] = 1  # King
    tensor[6, 0:8, 6] = 1  # Pawns

    # Convert tensor to board
    board = tensor_to_board(tensor)

    # Verify board state
    assert board.fen().split()[0] == "rrrqkrrr/pppppppp/8/8/8/8/PPPPPPPP/RRRQKRRR"
    print("Tensor to board conversion test passed!")


if __name__ == "__main__":
    test_tensor_to_board()
