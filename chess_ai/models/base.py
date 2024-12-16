from typing import Protocol, runtime_checkable

import torch
from chess import Board, Color, Move
from torch import Tensor

from chess_ai.data.preprocessing import GameEncoder
from chess_ai.utils.chess_utils import decode_move_index


@runtime_checkable
class ChessPolicyModel(Protocol):
    """Protocol defining the interface for chess policy models.

    Any class implementing this protocol must:
    1. Implement forward() with the specified signature
    2. Implement parameter_count()

    The @runtime_checkable decorator allows isinstance() checks,
    though they should be used sparingly in production code.
    """

    def forward(self, board_state: Tensor, additional_params: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            board_state: Tensor of shape (batch_size, channels, 8, 8)
                representing the chess board state
            additional_params: Tensor of shape (batch_size, num_params)
                containing additional features like castling rights

        Returns:
            Tensor of shape (batch_size, 4864) representing move probabilities
        """
        ...

    def parameter_count(self) -> int:
        """Returns the total number of parameters in the model."""
        ...

    def get_move(
        self,
        encoder: GameEncoder,
        board: Board,
        device: torch.device,
        color: Color,
        verbose: bool = False,
    ) -> Move: ...


@runtime_checkable
class ChessValueModel(Protocol):
    """Protocol defining the interface for chess value models.

    Any class implementing this protocol must:
    1. Implement forward() with the specified signature
    2. Implement parameter_count()

    The @runtime_checkable decorator allows isinstance() checks,
    though they should be used sparingly in production code.
    """

    def forward(self, board_state: Tensor, additional_params: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            board_state: Tensor of shape (batch_size, channels, 8, 8)
                representing the chess board state
            additional_params: Tensor of shape (batch_size, num_params)
                containing additional features like castling rights

        Returns:
            Tensor of shape (batch_size, 1) representing the value of the boards
        """
        ...

    def parameter_count(self) -> int:
        """Returns the total number of parameters in the model."""
        ...

    def __call__(self, state: Tensor, additional_params: Tensor) -> Tensor: ...

    def freeze_all_but_head(self): ...


def get_move(
    self: ChessPolicyModel,
    encoder: GameEncoder,
    board: Board,
    device: torch.device,
    color: Color,
    verbose: bool = False,
) -> Move:
    self.eval()
    with torch.no_grad():
        state, consts = encoder.encode_game_state(board)
        state = state.permute(2, 0, 1).unsqueeze(0).to(device)
        consts = consts.unsqueeze(0).to(device)
        outputs = self(state, consts)
        probabilities = torch.softmax(outputs, dim=1)
        _, sorted_indices = torch.sort(probabilities, descending=True)
        # Try moves in order of probability until we find a legal one
        for idx in sorted_indices[0]:
            # Decode move index to chess move
            # This will raise an error if the move is invalid
            # If the error is raised, we'll move on to the next move
            # implement error handling
            try:
                move = decode_move_index(int(idx.item()), color)
            except ValueError as e:
                if verbose:
                    print(f"Trying move: {idx.item()}; Error: {e}")
                continue
            # Why was it here?
            # if idx.item() == 4771:
            #     continue
            if verbose:
                print(f"Trying move {move.uci()} ({idx.item()})", end="")
            if move in board.legal_moves:
                if verbose:
                    print()
                return move
            if verbose:
                print(" - illegal")
    if verbose:
        print("No valid move found!")
    # Fallback to random legal move if no valid move found
    return next(iter(board.legal_moves))
