import chess
from torch import Tensor, zeros


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


def info_to_move(from_square: int, to_square: int, promotion=None):
    if from_square < 0 or from_square > 63 or to_square < 0 or to_square > 63:
        from_col = from_square % 8
        from_row = from_square // 8
        to_col = to_square % 8
        to_row = to_square // 8
        raise ValueError(f"Invalid move, out of bounds: {from_col}, {from_row} -> {to_col}, {to_row}")
    move = chess.Move(from_square, to_square)
    if promotion:
        if promotion == chess.QUEEN:
            move.promotion = chess.QUEEN
        elif promotion == chess.BISHOP:
            move.promotion = chess.BISHOP
        elif promotion == chess.ROOK:
            move.promotion = chess.ROOK
        elif promotion == chess.KNIGHT:
            move.promotion = chess.KNIGHT
    return move


def _slow_decode_move_index(index: int, color: chess.Color) -> chess.Move:
    tensor_8_8_76_form = zeros(8 * 8 * 76)
    tensor_8_8_76_form[index] = 1
    tensor_8_8_76_form = tensor_8_8_76_form.view(8, 8, 76)
    for row in range(8):
        for col in range(8):
            promotion = None
            from_square = chess.square(col, row)
            for i in range(64, 67):
                if tensor_8_8_76_form[row][col][i]:
                    promotion = chess.KNIGHT
                    to_square = chess.square(
                        col - 65 + i, row + 1 if color == chess.WHITE else row - 1
                    )
                    return info_to_move(from_square, to_square, promotion)
            for i in range(67, 70):
                if tensor_8_8_76_form[row][col][i]:
                    promotion = chess.ROOK
                    to_square = chess.square(
                        col - 68 + i, row + 1 if color == chess.WHITE else row - 1
                    )
                    return info_to_move(from_square, to_square, promotion)
            for i in range(70, 73):
                if tensor_8_8_76_form[row][col][i]:
                    promotion = chess.BISHOP
                    to_square = chess.square(
                        col - 71 + i, row + 1 if color == chess.WHITE else row - 1
                    )
                    return info_to_move(from_square, to_square, promotion)
            for i in range(73, 76):
                if tensor_8_8_76_form[row][col][i]:
                    promotion = chess.QUEEN
                    to_square = chess.square(
                        col - 74 + i, row + 1 if color == chess.WHITE else row - 1
                    )
                    return info_to_move(from_square, to_square, promotion)
            for i in range(7, 56, 8):
                if tensor_8_8_76_form[row][col][i]:
                    to_square = chess.square(col + i // 8 + 1, row + i // 8 + 1)
                    return info_to_move(from_square, to_square)
            for i in range(3, 52, 8):
                if tensor_8_8_76_form[row][col][i]:
                    to_square = chess.square(col - i // 8 - 1, row - i // 8 - 1)
                    return info_to_move(from_square, to_square)
            for i in range(1, 50, 8):
                if tensor_8_8_76_form[row][col][i]:
                    to_square = chess.square(col - i // 8 - 1, row + i // 8 + 1)
                    return info_to_move(from_square, to_square)
            for i in range(5, 54, 8):
                if tensor_8_8_76_form[row][col][i]:
                    to_square = chess.square(col + i // 8 + 1, row - i // 8 - 1)
                    return info_to_move(from_square, to_square)
            for i in range(6, 55, 8):
                if tensor_8_8_76_form[row][col][i]:
                    to_square = chess.square(col + i // 8 + 1, row)
                    return info_to_move(from_square, to_square)
            for i in range(2, 51, 8):
                if tensor_8_8_76_form[row][col][i]:
                    to_square = chess.square(col - i // 8 - 1, row)
                    return info_to_move(from_square, to_square)
            for i in range(0, 49, 8):
                if tensor_8_8_76_form[row][col][i]:
                    to_square = chess.square(col, row + i // 8 + 1)
                    return info_to_move(from_square, to_square)
            for i in range(4, 53, 8):
                if tensor_8_8_76_form[row][col][i]:
                    to_square = chess.square(col, row - i // 8 - 1)
                    return info_to_move(from_square, to_square)
            if tensor_8_8_76_form[row][col][56]:
                to_square = chess.square(col - 1, row + 2)
                return info_to_move(from_square, to_square)
            if tensor_8_8_76_form[row][col][57]:
                to_square = chess.square(col - 2, row + 1)
                return info_to_move(from_square, to_square)
            if tensor_8_8_76_form[row][col][58]:
                to_square = chess.square(col - 2, row - 1)
                return info_to_move(from_square, to_square)
            if tensor_8_8_76_form[row][col][59]:
                to_square = chess.square(col - 1, row - 2)
                return info_to_move(from_square, to_square)
            if tensor_8_8_76_form[row][col][60]:
                to_square = chess.square(col + 1, row - 2)
                return info_to_move(from_square, to_square)
            if tensor_8_8_76_form[row][col][61]:
                to_square = chess.square(col + 2, row - 1)
                return info_to_move(from_square, to_square)
            if tensor_8_8_76_form[row][col][62]:
                to_square = chess.square(col + 2, row + 1)
                return info_to_move(from_square, to_square)
            if tensor_8_8_76_form[row][col][63]:
                to_square = chess.square(col + 1, row + 2)
                return info_to_move(from_square, to_square)
    raise ValueError(f"Invalid move index: {index}")


def decode_move_index(index: int, color: chess.Color) -> chess.Move:
    """Convert a move index back to a chess move.

    Args:
        index: Move index from model output (0-4863)
        color: Color of the moving piece (for promotion direction)

    Returns:
        Corresponding chess move.

    Raises:
        ValueError: If the move index results in an invalid move.
    """
    move = index % 76
    col = (index // 76) % 8
    row = index // (8 * 76)
    from_square = chess.square(col, row)
    promotion = None
    if 64 <= move < 67:
        promotion = chess.KNIGHT
        to_square = chess.square(
            col - 65 + move, row + 1 if color == chess.WHITE else row - 1
        )
        return info_to_move(from_square, to_square, promotion)
    if 67 <= move < 70:
        promotion = chess.ROOK
        to_square = chess.square(
            col - 68 + move, row + 1 if color == chess.WHITE else row - 1
        )
        return info_to_move(from_square, to_square, promotion)
    if 70 <= move < 73:
        promotion = chess.BISHOP
        to_square = chess.square(
            col - 71 + move, row + 1 if color == chess.WHITE else row - 1
        )
        return info_to_move(from_square, to_square, promotion)
    if 73 <= move < 76:
        promotion = chess.QUEEN
        to_square = chess.square(
            col - 74 + move, row + 1 if color == chess.WHITE else row - 1
        )
        return info_to_move(from_square, to_square, promotion)
    if move % 8 == 7 and move < 56:
        to_square = chess.square(col + move // 8 + 1, row + move // 8 + 1)
        return info_to_move(from_square, to_square)
    if move % 8 == 3 and move < 52:
        to_square = chess.square(col - move // 8 - 1, row - move // 8 - 1)
        return info_to_move(from_square, to_square)
    if move % 8 == 1 and move < 50:
        to_square = chess.square(col - move // 8 - 1, row + move // 8 + 1)
        return info_to_move(from_square, to_square)
    if move % 8 == 5 and move < 54:
        to_square = chess.square(col + move // 8 + 1, row - move // 8 - 1)
        return info_to_move(from_square, to_square)
    if move % 8 == 6 and move < 55:
        to_square = chess.square(col + move // 8 + 1, row)
        return info_to_move(from_square, to_square)
    if move % 8 == 2 and move < 51:
        to_square = chess.square(col - move // 8 - 1, row)
        return info_to_move(from_square, to_square)
    if move % 8 == 0 and move < 49:
        to_square = chess.square(col, row + move // 8 + 1)
        return info_to_move(from_square, to_square)
    if move % 8 == 4 and move < 53:
        to_square = chess.square(col, row - move // 8 - 1)
        return info_to_move(from_square, to_square)
    if move == 56:
        to_square = chess.square(col - 1, row + 2)
        return info_to_move(from_square, to_square)
    if move == 57:
        to_square = chess.square(col - 2, row + 1)
        return info_to_move(from_square, to_square)
    if move == 58:
        to_square = chess.square(col - 2, row - 1)
        return info_to_move(from_square, to_square)
    if move == 59:
        to_square = chess.square(col - 1, row - 2)
        return info_to_move(from_square, to_square)
    if move == 60:
        to_square = chess.square(col + 1, row - 2)
        return info_to_move(from_square, to_square)
    if move == 61:
        to_square = chess.square(col + 2, row - 1)
        return info_to_move(from_square, to_square)
    if move == 62:
        to_square = chess.square(col + 2, row + 1)
        return info_to_move(from_square, to_square)
    if move == 63:
        to_square = chess.square(col + 1, row + 2)
        return info_to_move(from_square, to_square)
    raise ValueError(f"Invalid move index: {index}")
