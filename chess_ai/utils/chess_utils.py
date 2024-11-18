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
    # Get source square coordinates
    from_row = index // (8 * 76)  # Each square has 76 possible moves
    from_col = (index % (8 * 76)) // 76

    # Ensure the source square is within bounds
    if not (0 <= from_row < 8 and 0 <= from_col < 8):
        raise ValueError("Invalid source square: out of bounds")

    from_square = chess.square(from_col, from_row)
    move_type = index % 76

    # Handle queen moves (first 56 indices)
    if move_type < 56:
        direction = move_type % 8  # 8 possible directions
        distance = move_type // 8 + 1  # 1-7 squares in each direction

        # Destination square calculation
        if direction == 0:  # North
            to_col = from_col
            to_row = from_row + distance
        elif direction == 1:  # Northeast
            to_col = from_col + distance
            to_row = from_row + distance
        elif direction == 2:  # East
            to_col = from_col + distance
            to_row = from_row
        elif direction == 3:  # Southeast
            to_col = from_col + distance
            to_row = from_row - distance
        elif direction == 4:  # South
            to_col = from_col
            to_row = from_row - distance
        elif direction == 5:  # Southwest
            to_col = from_col - distance
            to_row = from_row - distance
        elif direction == 6:  # West
            to_col = from_col - distance
            to_row = from_row
        else:  # Northwest
            to_col = from_col - distance
            to_row = from_row + distance

        # Ensure the destination square is within bounds
        if not (0 <= to_row < 8 and 0 <= to_col < 8):
            raise ValueError("Invalid move: destination out of bounds")

        to_square = chess.square(to_col, to_row)
        return chess.Move(from_square, to_square)

    # Handle knight moves (next 8 indices)
    elif move_type < 64:
        knight_offset = move_type - 56
        knight_moves = [
            (-2, 1),
            (-1, 2),
            (1, 2),
            (2, 1),
            (2, -1),
            (1, -2),
            (-1, -2),
            (-2, -1),
        ]
        col_offset, row_offset = knight_moves[knight_offset]
        to_col = from_col + col_offset
        to_row = from_row + row_offset

        # Ensure the destination square is within bounds
        if not (0 <= to_row < 8 and 0 <= to_col < 8):
            raise ValueError("Invalid move: destination out of bounds")

        to_square = chess.square(to_col, to_row)
        return chess.Move(from_square, to_square)

    # Handle promotion moves (last 12 indices)
    else:
        promotion_type = (move_type - 64) // 3
        col_offset = (move_type - 64) % 3 - 1  # -1, 0, or 1
        promotion_pieces = [chess.KNIGHT, chess.ROOK, chess.BISHOP, chess.QUEEN]

        # Determine promotion direction based on pawn color
        row_offset = 1 if color == chess.WHITE else -1
        to_col = from_col + col_offset
        to_row = from_row + row_offset

        # Ensure the destination square is within bounds
        if not (0 <= to_row < 8 and 0 <= to_col < 8):
            raise ValueError("Invalid promotion move: destination out of bounds")

        to_square = chess.square(to_col, to_row)
        return chess.Move(
            from_square, to_square, promotion=promotion_pieces[promotion_type]
        )
