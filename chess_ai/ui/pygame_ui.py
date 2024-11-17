import os

import chess
import pygame

from .base import ChessUI, MoveResult


class PygameUI(ChessUI):
    """PyGame-based interface for chess."""

    SQUARE_SIZE = 80
    BOARD_SIZE = SQUARE_SIZE * 8
    WINDOW_SIZE = (BOARD_SIZE, BOARD_SIZE)

    def __init__(self):
        """Initialize PyGame UI."""
        pygame.init()
        self.screen = pygame.display.set_mode(self.WINDOW_SIZE)
        pygame.display.set_caption("Chess AI")

        # Load piece images
        self.pieces = {}
        for color in ["w", "b"]:
            for piece in ["p", "n", "b", "r", "q", "k"]:
                img_path = os.path.join("assets", "pieces", f"{color}{piece}.png")
                img = pygame.image.load(img_path)
                img = pygame.transform.scale(img, (self.SQUARE_SIZE, self.SQUARE_SIZE))
                self.pieces[f"{color}{piece}"] = img

        # Colors
        self.WHITE = (238, 238, 210)
        self.BLACK = (118, 150, 86)
        self.HIGHLIGHT = (186, 202, 43)

        self.selected_square = None
        self.running = True

    def display_board(self, board: chess.Board) -> None:
        """Display the current board state."""
        self.screen.fill((255, 255, 255))

        # Draw squares
        for row in range(8):
            for col in range(8):
                color = self.WHITE if (row + col) % 2 == 0 else self.BLACK
                rect = pygame.Rect(
                    col * self.SQUARE_SIZE,
                    (7 - row) * self.SQUARE_SIZE,
                    self.SQUARE_SIZE,
                    self.SQUARE_SIZE,
                )
                pygame.draw.rect(self.screen, color, rect)

                # Draw piece if present
                square = chess.square(col, row)
                piece = board.piece_at(square)
                if piece:
                    color = "w" if piece.color else "b"
                    piece_name = f"{color}{piece.symbol().lower()}"
                    piece_img = self.pieces[piece_name]
                    piece_pos = (col * self.SQUARE_SIZE, (7 - row) * self.SQUARE_SIZE)
                    self.screen.blit(piece_img, piece_pos)

        # Highlight selected square
        if self.selected_square is not None:
            col = chess.square_file(self.selected_square)
            row = chess.square_rank(self.selected_square)
            rect = pygame.Rect(
                col * self.SQUARE_SIZE,
                (7 - row) * self.SQUARE_SIZE,
                self.SQUARE_SIZE,
                self.SQUARE_SIZE,
            )
            pygame.draw.rect(self.screen, self.HIGHLIGHT, rect, 3)

        pygame.display.flip()

    def get_player_move(self, board: chess.Board) -> chess.Move | None:
        """Get a move from the player via mouse input."""
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return None

                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    file = x // self.SQUARE_SIZE
                    rank = 7 - (y // self.SQUARE_SIZE)
                    square = chess.square(file, rank)

                    if self.selected_square is None:
                        piece = board.piece_at(square)
                        if piece and piece.color == board.turn:
                            self.selected_square = square
                            self.display_board(board)
                    else:
                        move = chess.Move(self.selected_square, square)
                        self.selected_square = None
                        if move in board.legal_moves:
                            return move
                        self.display_board(board)

    def show_move_result(self, result: MoveResult) -> None:
        """Display the result of a move."""
        if not result.success:
            print(f"Error: {result.message}")
        elif result.message:
            print(result.message)

    def show_game_end(self, winner: str | None) -> None:
        """Display game end state."""
        font = pygame.font.Font(None, 74)
        if winner is None:
            text = font.render("Draw!", True, (0, 0, 0))
        else:
            text = font.render(f"{winner.capitalize()} wins!", True, (0, 0, 0))

        text_rect = text.get_rect(center=(self.BOARD_SIZE // 2, self.BOARD_SIZE // 2))
        overlay = pygame.Surface(self.WINDOW_SIZE, pygame.SRCALPHA)
        overlay.fill((255, 255, 255, 128))
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text, text_rect)
        pygame.display.flip()

        # Wait for a moment before closing
        pygame.time.wait(3000)
        pygame.quit()
