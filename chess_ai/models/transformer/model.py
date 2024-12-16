import torch
import torch.nn as nn
from chess import Board, Color, Move

from chess_ai.data.preprocessing import GameEncoder
from chess_ai.models.base import ChessPolicyModel, get_move


class ChessTransformer(nn.Module):
    """Transformer-based chess embedding model."""

    def __init__(
        self,
        d_model: int = 16,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 32,
        dropout: float = 0.1,
    ) -> None:
        """Initialize transformer model.

        Args:
            d_model: Dimension of model embeddings
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
        """
        super().__init__()

        # Embed board positions (8x8 = 64 positions)
        self.pos_embedding = nn.Embedding(64, d_model)

        # Embed piece types (12 channels: 6 pieces x 2 colors)
        self.piece_embedding = nn.Embedding(
            13, d_model
        )  # 12 piece types + empty square

        # Additional features embedding
        self.additional_embedding = nn.Linear(3, d_model)

        # Transformer blocks
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

    def forward(
        self,
        board_state: torch.Tensor,
        additional_params: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            board_state: Tensor of shape (batch_size, channels, 8, 8)
            additional_params: Tensor of shape (batch_size, 3)
            mask: Optional mask for transformer

        Returns:
            Tensor of shape (batch_size, 4864) representing move probabilities
        """
        batch_size = board_state.size(0)

        # Create position indices
        pos_indices = torch.arange(64, device=board_state.device)
        pos_indices = pos_indices.expand(batch_size, -1)

        # Embed positions
        pos_embeddings = self.pos_embedding(pos_indices)  # (batch, 64, d_model)

        # Process board state
        board_flat = board_state.view(batch_size, 12, 64).transpose(
            1, 2
        )  # (batch, 64, 12)
        piece_indices = torch.argmax(board_flat, dim=2)  # Convert one-hot to indices
        piece_embeddings = self.piece_embedding(piece_indices)  # (batch, 64, d_model)

        # Combine position and piece embeddings
        square_features = pos_embeddings + piece_embeddings

        # Process additional parameters
        additional_features = self.additional_embedding(
            additional_params.float()
        )  # (batch, d_model)
        additional_features = additional_features.unsqueeze(1)  # (batch, 1, d_model)

        # Combine all features
        encoder_input = torch.cat(
            [square_features, additional_features], dim=1
        )  # (batch, 65, d_model)

        # Single transformer pass
        output = self.transformer(encoder_input, mask)

        return output


class TransformerPolicyModel(nn.Module):
    """A transformer-based chess policy model."""

    def __init__(
        self,
        d_model: int = 16,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 32,
        dim_decoder: int = 4864,
        dropout: float = 0.1,
    ) -> None:
        """Initialize transformer policy model.

        Args:
            d_model: Dimension of model embeddings
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dim_decoder: Dimension of decoder network
            dropout: Dropout rate
        """
        super().__init__()
        self.encoder = ChessTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.move_predictor = nn.Sequential(
            nn.Linear(d_model * 65, dim_decoder),  # 64 squares + additional features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_decoder, 4864),  # Total possible moves
        )

    def forward(
        self, board_state: torch.Tensor, additional_params: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            board_state: Tensor of shape (batch_size, 12, 8, 8)
            additional_params: Tensor of shape (batch_size, 3)

        Returns:
            Tensor of shape (batch_size, 4864) representing move probabilities
        """
        batch_size = board_state.size(0)
        output = self.encoder(board_state, additional_params)

        output_flat = output.reshape(batch_size, -1)
        move_logits = self.move_predictor(output_flat)

        return move_logits

    def parameter_count(self) -> int:
        """Returns the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def get_move(
        self: ChessPolicyModel,
        encoder: GameEncoder,
        board: Board,
        device: torch.device,
        color: Color,
        verbose: bool = False,
    ) -> Move:
        return get_move(self, encoder, board, device, color, verbose)


class TransformerValueModel(nn.Module):
    """A transformer-based chess value model."""

    def __init__(
        self,
        d_model: int = 16,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 32,
        dim_decoder: int = 1,
        dropout: float = 0.1,
    ) -> None:
        """Initialize transformer value model.

        Args:
            d_model: Dimension of model embeddings
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dim_decoder: Dimension of decoder network
            dropout: Dropout rate
        """
        super().__init__()
        self.encoder = ChessTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.value_predictor = nn.Sequential(
            nn.Linear(d_model * 65, dim_decoder),  # 64 squares + additional features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_decoder, 1),  # Single value output
        )

    def forward(
        self, board_state: torch.Tensor, additional_params: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            board_state: Tensor of shape (batch_size, 12, 8, 8)
            additional_params: Tensor of shape (batch_size, 3)

        Returns:
            Tensor of shape (batch_size, 1) representing value
        """
        batch_size = board_state.size(0)
        output = self.encoder(board_state, additional_params)

        output_flat = output.reshape(batch_size, -1)
        value = self.value_predictor(output_flat)

        return value

    def parameter_count(self) -> int:
        """Returns the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def get_move(
        self: ChessPolicyModel,
        encoder: GameEncoder,
        board: Board,
        device: torch.device,
        color: Color,
        verbose: bool = False,
    ) -> Move:
        return get_move(self, encoder, board, device, color, verbose)

    @classmethod
    def initialize_from_policy(
        cls,
        policy_model_path: str,
        d_model: int = 16,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 32,
        dim_decoder: int = 1,
        dropout: float = 0.1,
    ) -> "TransformerValueModel":
        """Initialize a value model from a policy model."""
        value_model = cls(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dim_decoder=dim_decoder,
            dropout=dropout,
        )
        policy_model = cls(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dim_decoder=4864,
            dropout=dropout,
        )

        policy_model.load_state_dict(torch.load(policy_model_path))
        value_model.encoder = policy_model.encoder
        return value_model
