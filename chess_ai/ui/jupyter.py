from typing import Callable

import chess
from IPython.display import HTML, Javascript, display

from .base import ChessUI, MoveResult


class JupyterUI(ChessUI):
    """Chess UI implementation for Jupyter notebooks."""

    def __init__(self):
        """Initialize the UI."""
        self._init_chessboard_js()
        self.current_move: chess.Move | None = None
        self.board: chess.Board | None = None
        self._move_callback: Callable[[str, str, str | None], None] | None = None

    def display_board(self, board: chess.Board) -> None:
        """Display the current board state."""
        self.board = board
        self._update_board_js(board.fen())

    def get_player_move(self, board: chess.Board) -> chess.Move | None:
        """Get a move from the player using the interactive board."""
        import time

        self.board = board
        self.current_move = None

        # Set up the move callback
        def move_callback(
            source: str, target: str, promotion: str | None = None
        ) -> None:
            try:
                move = chess.Move.from_uci(f"{source}{target}{promotion or ''}")
                if move in board.legal_moves:
                    self.current_move = move
            except ValueError:
                pass

        self._move_callback = move_callback

        # Wait for move
        while self.current_move is None:
            time.sleep(0.1)  # Prevent busy waiting

        move = self.current_move
        self.current_move = None
        return move

    def show_move_result(self, result: MoveResult) -> None:
        """Display the result of a move."""
        style = "color: red;" if not result.success else ""
        display(HTML(f"<p style='{style}'>{result.message}</p>"))

    def show_game_end(self, winner: str | None) -> None:
        """Display game end state."""
        if winner is None:
            message = "Game ended in a draw!"
        else:
            message = f"Game over - {winner.capitalize()} wins!"

        display(HTML(f"<h3>{message}</h3>"))

    def _init_chessboard_js(self) -> None:
        """Initialize the chessboard.js board."""
        js_code = """
        <link rel="stylesheet"
              href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css"
              integrity="sha384-q94+BZtLrkL1/ohfjR8c6L+A6qzNH9R2hBLwyoAfu3i/WCvQjzL2RQJ3uNHDISdU"
              crossorigin="anonymous">

        <script src="https://code.jquery.com/jquery-3.5.1.min.js"
                integrity="sha384-ZvpUoO/+PpLXR1lu4jmpXWu80pZlYUAfxl5NsBMWOEPSjUn/6Z/hRTt8+pR6L4N2"
                crossorigin="anonymous"></script>

        <script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"
                integrity="sha384-8Vi8VHwn3vjQ9eUHUxex3JSN/NFqUg3QbPyX8kWyb93+8AC/pPWTzj+nHtbC5bxD"
                crossorigin="anonymous"></script>

        <div id="board1" style="width: 400px"></div>
        <div id="promotion-dialog" style="display: none;">
            <p>Choose promotion piece:</p>
            <button onclick="selectPromotion('q')">Queen</button>
            <button onclick="selectPromotion('r')">Rook</button>
            <button onclick="selectPromotion('b')">Bishop</button>
            <button onclick="selectPromotion('n')">Knight</button>
        </div>
        <p id="status"></p>

        <script>
        var board = null;
        var $status = $('#status');
        var currentSource = null;
        var currentTarget = null;
        var isPromoting = false;

        function onDragStart (source, piece, position, orientation) {
            // Only pick up pieces for the side to move
            if ((orientation === 'white' && piece.search(/^b/) !== -1) ||
                (orientation === 'black' && piece.search(/^w/) !== -1)) {
                return false;
            }
        }

        function showPromotionDialog() {
            $('#promotion-dialog').show();
            isPromoting = true;
        }

        function selectPromotion(piece) {
            $('#promotion-dialog').hide();
            isPromoting = false;

            // Send move to Python with promotion piece
            (async function() {
                const result = await google.colab.kernel.invokeFunction(
                    'notebook.move_callback',
                    [currentSource, currentTarget, piece],
                    {}
                );
            })();
        }

        function onDrop (source, target, piece, newPos, oldPos, orientation) {
            // Store current move
            currentSource = source;
            currentTarget = target;

            // Check if this is a pawn promotion move
            if ((piece === 'wP' && target[1] === '8') ||
                (piece === 'bP' && target[1] === '1')) {
                showPromotionDialog();
                return;
            }

            // Send move to Python
            (async function() {
                const result = await google.colab.kernel.invokeFunction(
                    'notebook.move_callback',
                    [source, target, null],
                    {}
                );
            })();
        }

        var config = {
            draggable: true,
            position: 'start',
            onDragStart: onDragStart,
            onDrop: onDrop,
            pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png'
        };

        board = Chessboard('board1', config);

        // Function to update the board position
        function updateBoard(fen) {
            board.position(fen);
        }
        </script>
        """
        display(HTML(js_code))

        # Register Python callback for move handling
        from google.colab import output

        output.register_callback(
            "notebook.move_callback",
            lambda source, target, promotion: self._handle_move(
                source, target, promotion
            ),
        )

    def _update_board_js(self, fen: str) -> None:
        """Update the board position."""
        js_code = f"updateBoard('{fen}')"
        display(Javascript(js_code))

    def _handle_move(self, source: str, target: str, promotion: str | None) -> None:
        """Handle move from JavaScript."""
        if self._move_callback:
            self._move_callback(source, target, promotion)
