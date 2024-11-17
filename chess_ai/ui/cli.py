import chess

from .base import ChessUI, MoveResult


class CliUI(ChessUI):
    """Simple command-line interface for chess."""

    def display_board(self, board: chess.Board) -> None:
        """Display the current board state."""
        print("\n" + str(board) + "\n")

    def get_player_move(self, board: chess.Board) -> chess.Move | None:
        """Get a move from the player via command line input."""
        while True:
            try:
                move_str = input("Enter your move (e.g. e2e4) or 'q' to quit: ")
                if move_str.lower() == "q":
                    return None

                move = chess.Move.from_uci(move_str)
                if move in board.legal_moves:
                    return move
                else:
                    print("Illegal move! Try again.")
            except ValueError:
                print("Invalid move format! Use format like 'e2e4'")
            except KeyboardInterrupt:
                return None

    def show_move_result(self, result: MoveResult) -> None:
        """Display the result of a move."""
        if not result.success:
            print(f"Error: {result.message}")
        elif result.message:
            print(result.message)

    def show_game_end(self, winner: str | None) -> None:
        """Display game end state."""
        print("\n=== Game Over ===")
        if winner is None:
            print("Game ended in a draw!")
        else:
            print(f"{winner.capitalize()} wins!")

