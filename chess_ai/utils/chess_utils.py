import chess
from torch import Tensor


def tensor_to_board(tensor: Tensor) -> chess.Board:
    """Convert a board tensor back to a chess.Board.

    Args:
        tensor: Tensor of shape (8, 8, 12) representing the board state

    Returns:
        chess.Board with the corresponding position
    """
    board = chess.Board()
    board.clear()

    idx_to_piece = [
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    ]

    for row in range(8):
        for col in range(8):
            for piece_idx in range(6):
                # Check white pieces
                white_offset = 0
                piece_is_here = tensor[row][col][piece_idx + white_offset]
                if piece_is_here:
                    piece = idx_to_piece[piece_idx]
                    board.set_piece_at(
                        chess.square(col, row), chess.Piece(piece, chess.WHITE)
                    )

                # Check black pieces
                black_offset = 6
                piece_is_here = tensor[row][col][piece_idx + black_offset]
                if piece_is_here:
                    piece = idx_to_piece[piece_idx]
                    board.set_piece_at(
                        chess.square(col, row), chess.Piece(piece, chess.BLACK)
                    )

    return board


def tensor_to_move(tensor: Tensor, color: bool) -> chess.Move | None:
    """Convert a move tensor back to a chess.Move."""
    tensor_8_8_76_form = tensor.view(8, 8, 76)

    def info_to_move(
        from_square: int, to_square: int, promotion: int | None = None
    ) -> chess.Move:
        move = chess.Move(from_square, to_square)
        if promotion:
            move.promotion = promotion
        return move

    for row in range(8):
        for col in range(8):
            # Promotion moves
            for i in range(64, 67):  # Knight promotions
                if tensor_8_8_76_form[row][col][i]:
                    to_square = chess.square(
                        col - 65 + i, row + 1 if color == chess.WHITE else row - 1
                    )
                    return info_to_move(chess.square(col, row), to_square, chess.KNIGHT)

            for i in range(67, 70):  # Rook promotions
                if tensor_8_8_76_form[row][col][i]:
                    to_square = chess.square(
                        col - 68 + i, row + 1 if color == chess.WHITE else row - 1
                    )
                    return info_to_move(chess.square(col, row), to_square, chess.ROOK)

            for i in range(70, 73):  # Bishop promotions
                if tensor_8_8_76_form[row][col][i]:
                    to_square = chess.square(
                        col - 71 + i, row + 1 if color == chess.WHITE else row - 1
                    )
                    return info_to_move(chess.square(col, row), to_square, chess.BISHOP)

            for i in range(73, 76):  # Queen promotions
                if tensor_8_8_76_form[row][col][i]:
                    to_square = chess.square(
                        col - 74 + i, row + 1 if color == chess.WHITE else row - 1
                    )
                    return info_to_move(chess.square(col, row), to_square, chess.QUEEN)

            # Regular moves
            # Diagonal moves
            for i in range(7, 56, 8):  # Up-right diagonal
                if tensor_8_8_76_form[row][col][i]:
                    to_square = chess.square(col + i // 8 + 1, row + i // 8 + 1)
                    return info_to_move(chess.square(col, row), to_square)

            for i in range(3, 52, 8):  # Down-left diagonal
                if tensor_8_8_76_form[row][col][i]:
                    to_square = chess.square(col - i // 8 - 1, row - i // 8 - 1)
                    return info_to_move(chess.square(col, row), to_square)

            for i in range(1, 50, 8):  # Up-left diagonal
                if tensor_8_8_76_form[row][col][i]:
                    to_square = chess.square(col - i // 8 - 1, row + i // 8 + 1)
                    return info_to_move(chess.square(col, row), to_square)

            for i in range(5, 54, 8):  # Down-right diagonal
                if tensor_8_8_76_form[row][col][i]:
                    to_square = chess.square(col + i // 8 + 1, row - i // 8 - 1)
                    return info_to_move(chess.square(col, row), to_square)

            # Straight moves
            for i in range(6, 55, 8):  # Right
                if tensor_8_8_76_form[row][col][i]:
                    to_square = chess.square(col + i // 8 + 1, row)
                    return info_to_move(chess.square(col, row), to_square)

            for i in range(2, 51, 8):  # Left
                if tensor_8_8_76_form[row][col][i]:
                    to_square = chess.square(col - i // 8 - 1, row)
                    return info_to_move(chess.square(col, row), to_square)

            for i in range(0, 49, 8):  # Up
                if tensor_8_8_76_form[row][col][i]:
                    to_square = chess.square(col, row + i // 8 + 1)
                    return info_to_move(chess.square(col, row), to_square)

            for i in range(4, 53, 8):  # Down
                if tensor_8_8_76_form[row][col][i]:
                    to_square = chess.square(col, row - i // 8 - 1)
                    return info_to_move(chess.square(col, row), to_square)

            # Knight moves
            if tensor_8_8_76_form[row][col][56]:  # Up 2, Left 1
                to_square = chess.square(col - 1, row + 2)
                return info_to_move(chess.square(col, row), to_square)

            if tensor_8_8_76_form[row][col][57]:  # Up 1, Left 2
                to_square = chess.square(col - 2, row + 1)
                return info_to_move(chess.square(col, row), to_square)

            if tensor_8_8_76_form[row][col][58]:  # Down 1, Left 2
                to_square = chess.square(col - 2, row - 1)
                return info_to_move(chess.square(col, row), to_square)

            if tensor_8_8_76_form[row][col][59]:  # Down 2, Left 1
                to_square = chess.square(col - 1, row - 2)
                return info_to_move(chess.square(col, row), to_square)

            if tensor_8_8_76_form[row][col][60]:  # Down 2, Right 1
                to_square = chess.square(col + 1, row - 2)
                return info_to_move(chess.square(col, row), to_square)

            if tensor_8_8_76_form[row][col][61]:  # Down 1, Right 2
                to_square = chess.square(col + 2, row - 1)
                return info_to_move(chess.square(col, row), to_square)

            if tensor_8_8_76_form[row][col][62]:  # Up 1, Right 2
                to_square = chess.square(col + 2, row + 1)
                return info_to_move(chess.square(col, row), to_square)

            if tensor_8_8_76_form[row][col][63]:  # Up 2, Right 1
                to_square = chess.square(col + 1, row + 2)
                return info_to_move(chess.square(col, row), to_square)

    return None


class EloRange:
    """Handles ELO rating ranges for dataset filtering."""

    RANGES = [800, 1200, 1600, 2000, 2400]
    MULTIPLIERS = [1, 2, 4, 4, 3]

    @classmethod
    def get_range_str(cls, base_rating: int, part: int | None = None) -> str:
        """Get string representation of rating range.

        Args:
            base_rating: Base rating (e.g., 1600)
            part: Optional part number for splitting range

        Returns:
            Rating range string (e.g., "1600-2000-2")
        """
        if base_rating not in cls.RANGES:
            raise ValueError(f"Invalid base rating. Must be one of {cls.RANGES}")

        range_str = f"{base_rating}-{base_rating + 400}"
        if part is not None:
            range_str += f"-{part}"
        return range_str

    @classmethod
    def get_num_games(cls, base_rating: int) -> int:
        """Get number of games to use for a rating range.

        Args:
            base_rating: Base rating

        Returns:
            Number of games to use
        """
        idx = cls.RANGES.index(base_rating)
        return 4000 * cls.MULTIPLIERS[idx]
