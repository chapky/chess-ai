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
        """Encodes a move into a move index.

        The move space is encoded as follows:
        - Regular moves: 0-4799
        - Promotions: 4800-4863
        """
        # Implementation of move encoding logic
        # This is a simplified version - you'll want to add the full logic
        base_idx = move.from_square * 64 + move.to_square
        if move.promotion:
            promotion_offset = {
                chess.QUEEN: 0,
                chess.ROOK: 16,
                chess.BISHOP: 32,
                chess.KNIGHT: 48,
            }[move.promotion]
            return 4800 + promotion_offset + (move.from_square % 8)
        return base_idx
