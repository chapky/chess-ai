import chess

from chess_ai.ui.jupyter import JupyterUI


def test_ui():
    # Initialize UI
    ui = JupyterUI()
    board = chess.Board()

    # Test board display
    try:
        ui.display_board(board)
        print("UI display test passed!")
    except Exception as e:
        print(f"UI display test failed: {str(e)}")

    # Test move handling
    try:
        ui._handle_move("e2", "e4", None)
        print("UI move handling test passed!")
    except Exception as e:
        print(f"UI move handling test failed: {str(e)}")


if __name__ == "__main__":
    test_ui()
