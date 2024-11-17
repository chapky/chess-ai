import torch

from chess_ai.models.cnn.model import ChessAISmaller


def test_model():
    # Initialize model
    model = ChessAISmaller()

    # Create sample input
    batch_size = 4
    board_tensor = torch.randn(batch_size, 12, 8, 8)  # Board state
    const_tensor = torch.randn(batch_size, 3)  # Additional parameters

    # Test forward pass
    try:
        output = model(board_tensor, const_tensor)
        print(f"Model output shape: {output.shape}")
        assert output.shape == (batch_size, 4864), "Unexpected output shape"
        print("Model test passed!")
    except Exception as e:
        print(f"Model test failed: {str(e)}")


if __name__ == "__main__":
    test_model()
