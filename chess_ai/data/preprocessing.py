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
        - Each source square (0-63) has 76 possible moves:
            - Queen moves (0-55): 8 directions × 7 distances
            - Knight moves (56-63): 8 possible knight moves
            - Promotions (64-75): 4 piece types × 3 files
        """
        from_sq = move.from_square
        from_row, from_col = divmod(from_sq, 8)
        to_row, to_col = divmod(move.to_square, 8)
        
        # Base index for the source square
        base_index = from_row * 8 * 76 + from_col * 76
        
        # Handle promotions
        if move.promotion:
            promotion_types = {
                chess.KNIGHT: 0,
                chess.ROOK: 1,
                chess.BISHOP: 2,
                chess.QUEEN: 3
            }
            col_offset = to_col - from_col + 1  # -1, 0, 1 -> 0, 1, 2
            promotion_idx = promotion_types[move.promotion] * 3 + col_offset
            return base_index + 64 + promotion_idx
        
        # Handle knight moves
        row_diff = to_row - from_row
        col_diff = to_col - from_col
        knight_moves = [
            (-2, 1), (-1, 2), (1, 2), (2, 1),
            (2, -1), (1, -2), (-1, -2), (-2, -1)
        ]
        if (row_diff, col_diff) in knight_moves:
            knight_idx = knight_moves.index((row_diff, col_diff))
            return base_index + 56 + knight_idx
        
        # Handle queen moves (including rook and bishop moves)
        distance = max(abs(row_diff), abs(col_diff))
        if row_diff == 0:  # Horizontal
            direction = 2 if col_diff > 0 else 6  # East or West
        elif col_diff == 0:  # Vertical
            direction = 0 if row_diff > 0 else 4  # North or South
        elif abs(row_diff) == abs(col_diff):  # Diagonal
            if row_diff > 0:  # North diagonals
                direction = 1 if col_diff > 0 else 7  # Northeast or Northwest
            else:  # South diagonals
                direction = 3 if col_diff > 0 else 5  # Southeast or Southwest
        else:
            raise ValueError("Invalid move: not a valid queen, rook, or bishop move")
        
        return base_index + (distance - 1) * 8 + direction
