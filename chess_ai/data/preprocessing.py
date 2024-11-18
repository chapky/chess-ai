from typing import Protocol

import chess
import torch
from torch import Tensor


class GameEncoder(Protocol):
    """Protocol for encoding chess games into tensors."""

    def encode_game_state(self, game: chess.Board) -> tuple[Tensor, Tensor]:
        """Convert a game state into board and constant tensors.

        Args:
            game: Current game state

        Returns:
            Tuple containing:
                - board_tensor: Tensor of shape (8, 8, 12)
                - const_tensor: Tensor of shape (3,) containing additional state
        """
        ...

    def encode_move(self, move: chess.Move) -> int:
        """Convert a chess move into a move index.

        Args:
            move: Chess move to encode

        Returns:
            Integer index representing the move (0-4863)
        """
        ...


class StandardEncoder:
    """Standard implementation of game encoding."""

    def encode_game_state(self, game: chess.Board) -> tuple[Tensor, Tensor]:
        """Encodes game state into tensors.

        The board is encoded as a 8x8x12 tensor where:
            - First 6 channels represent white pieces
            - Next 6 channels represent black pieces
            - Within each color, channels are: [pawn, knight, bishop, rook, queen, king]
        """
        tensor = torch.zeros(8, 8, 12)

        piece_to_idx = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5,
        }

        color = game.turn

        # Encode pieces
        for i in range(64):
            piece = game.piece_at(i)
            if piece:
                color_offset = 0 if piece.color == color else 6
                piece_idx = piece_to_idx[piece.piece_type]
                row, col = divmod(i, 8)
                tensor[row, col, color_offset + piece_idx] = 1

        # Encode additional state
        consts = torch.tensor(
            [
                bool(
                    game.castling_rights & (chess.BB_A1 if color else chess.BB_A8)
                ),  # queen-side
                bool(
                    game.castling_rights & (chess.BB_H1 if color else chess.BB_H8)
                ),  # king-side
                1 if game.turn == chess.WHITE else 0,  # current turn
            ]
        )

        return tensor, consts

    def encode_move(self, move: chess.Move) -> int:
        """Convert a chess move into a move index.

        The move space is encoded as:
        - Queen moves (0-55): 8 directions × 7 distances
        - Knight moves (56-63): 8 possible knight moves
        - Promotions (64-75): 4 piece types × 3 files
        """
        from_sq, to_sq = move.from_square, move.to_square
        from_row, from_col = divmod(from_sq, 8)
        to_row, to_col = divmod(to_sq, 8)

        # Handle promotions
        if move.promotion:
            col_offset = to_col - from_col + 1  # -1, 0, or 1 -> 0, 1, 2
            if move.promotion == chess.QUEEN:
                return 64 + 9 + col_offset
            elif move.promotion == chess.BISHOP:
                return 64 + 6 + col_offset
            elif move.promotion == chess.ROOK:
                return 64 + 3 + col_offset
            elif move.promotion == chess.KNIGHT:
                return 64 + col_offset

        # Handle knight moves
        row_diff = to_row - from_row
        col_diff = to_col - from_col
        if abs(row_diff) == 2 and abs(col_diff) == 1:
            if row_diff == 2:
                return 56 + (3 if col_diff == 1 else 4)
            else:  # row_diff == -2
                return 56 + (0 if col_diff == 1 else 7)
        elif abs(row_diff) == 1 and abs(col_diff) == 2:
            if row_diff == 1:
                return 56 + (1 if col_diff == 2 else 6)
            else:  # row_diff == -1
                return 56 + (2 if col_diff == 2 else 5)

        # Handle queen moves (including rook and bishop moves)
        distance = max(abs(row_diff), abs(col_diff)) - 1
        if row_diff == col_diff:  # Diagonal
            if row_diff > 0:  # Up-right
                return 7 + distance * 8
            else:  # Down-left
                return 3 + distance * 8
        elif row_diff == -col_diff:  # Anti-diagonal
            if row_diff > 0:  # Up-left
                return 1 + distance * 8
            else:  # Down-right
                return 5 + distance * 8
        elif row_diff == 0:  # Horizontal
            if col_diff > 0:  # Right
                return 6 + distance * 8
            else:  # Left
                return 2 + distance * 8
        else:  # Vertical
            if row_diff > 0:  # Up
                return 0 + distance * 8
            else:  # Down
                return 4 + distance * 8
