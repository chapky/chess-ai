from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class ChessPolicyModel(Protocol):
    """Protocol defining the interface for chess models.

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


# Example implementation check function
def check_model_implementation(model) -> bool:
    """Verify that a model properly implements the ChessPolicyModel protocol.

    Args:
        model: The model to check

    Returns:
        bool: True if the model implements the protocol correctly

    Raises:
        TypeError: If the model doesn't implement required methods
    """
    if not isinstance(model, ChessPolicyModel):
        raise TypeError(
            f"{model.__class__.__name__} doesn't implement the ChessPolicyModel protocol.\n"
            "Required methods: forward(board_state, additional_params), "
            "parameter_count()"
        )
    return True
