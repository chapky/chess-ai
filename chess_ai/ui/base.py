from dataclasses import dataclass
from typing import Protocol

import chess
import torch
from chess import pgn

from chess_ai.utils.chess_utils import decode_move_index


@dataclass
class MoveResult:
    """Contains the result of a move attempt."""

    success: bool
    message: str = ""
    game_over: bool = False
    winner: str | None = None


class ChessUI(Protocol):
    """Protocol for chess user interfaces."""

    def display_board(self, board: chess.Board) -> None:
        """Display the current board state.

        Args:
            board: Current chess board
        """
        ...

    def get_player_move(self, board: chess.Board) -> chess.Move | None:
        """Get a move from the player.

        Args:
            board: Current chess board

        Returns:
            The selected move or None if cancelled
        """
        ...

    def show_move_result(self, result: MoveResult) -> None:
        """Display the result of a move.

        Args:
            result: The move result to display
        """
        ...

    def show_game_end(self, winner: str | None) -> None:
        """Display game end state.

        Args:
            winner: The winner ('white', 'black', or None for draw)
        """
        ...


class GameController:
    """Controls the game flow and interfaces between UI and model."""

    def __init__(
        self,
        ui: ChessUI,
        model: torch.nn.Module,
        encoder,  # Using the encoder protocol from data/preprocessing.py
        device: torch.device,
        color: chess.Color = chess.WHITE,
    ):
        """Initialize controller.

        Args:
            ui: User interface implementation
            model: Trained chess model
            encoder: Game state encoder
            device: Device to run model on
        """
        self.ui = ui
        self.model = model
        self.encoder = encoder
        self.device = device
        self.board = chess.Board()
        self.color = color

    def play_game(self, player_color: chess.Color = chess.WHITE) -> None:
        """Play a full game.

        Args:
            player_color: Color for the human player
        """
        self.board.reset()
        game = pgn.Game.from_board(self.board)

        while not game.board().is_game_over():
            self.ui.display_board(game.board())

            if game.turn() == player_color:
                move = self.ui.get_player_move(game.board())
                if move is None:
                    return  # Game cancelled
            else:
                move = self._get_model_move(game.board())

            result = self._make_move(game, move)
            self.ui.show_move_result(result)

            if result.game_over:
                self.ui.show_game_end(result.winner)
                break

            game = game.next()

    def _get_model_move(self, game: chess.Board) -> chess.Move:
        """Get a move from the model.

        Args:
            game: Current game state

        Returns:
            Selected move
        """
        self.model.eval()
        with torch.no_grad():
            # Encode game state
            state, consts = self.encoder.encode_game_state(game)
            state = state.permute(2, 0, 1).unsqueeze(0).to(self.device)
            consts = consts.unsqueeze(0).to(self.device)

            # Get model prediction
            outputs = self.model(state, consts)
            probabilities = torch.softmax(outputs, dim=1)

            # Try moves in order of probability until we find a legal one
            _, sorted_indices = torch.sort(probabilities, descending=True)
            for idx in sorted_indices[0]:
                # Decode move index to chess move
                # This will raise an error if the move is invalid
                # If the error is raised, we'll move on to the next move
                # implement error handling
                try:
                    move = decode_move_index(int(idx.item()), self.color)
                except ValueError as e:
                    print(f"Error: {e}")
                    continue
                print(f"Trying move: {idx.item()}")
                if idx.item() == 4771:
                    continue
                if move in game.legal_moves:
                    return move
        print("No valid move found!")
        # Fallback to random legal move if no valid move found
        return next(iter(game.legal_moves))

    def _make_move(self, game: pgn.Game, move: chess.Move) -> MoveResult:
        """Attempt to make a move.

        Args:
            game: Current game state
            move: Move to make

        Returns:
            Result of the move attempt
        """
        if move not in game.board().legal_moves:
            return MoveResult(success=False, message="Illegal move")

        game.add_main_variation(move)

        if game.board().is_game_over():
            outcome = game.board().outcome()
            assert outcome is not None

            winner = None
            if outcome.winner == chess.WHITE:
                winner = "white"
            elif outcome.winner == chess.BLACK:
                winner = "black"

            return MoveResult(
                success=True,
                game_over=True,
                winner=winner,
                message=self._get_game_over_message(outcome),
            )

        return MoveResult(success=True, message=f"Move played: {move.uci()}")

    def _get_game_over_message(self, outcome: chess.Outcome) -> str:
        """Get a human-readable game over message.

        Args:
            outcome: Game outcome

        Returns:
            Message describing the game end
        """
        if outcome.winner is not None:
            winner = "White" if outcome.winner else "Black"
            return f"Game Over - {winner} wins by {outcome.termination.name}"
        return f"Game Over - Draw by {outcome.termination.name}"

