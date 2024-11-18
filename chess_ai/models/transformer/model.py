import torch
import torch.nn as nn


class ChessTransformer(nn.Module):
    """Transformer-based chess model."""

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
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

        # Output layers
        self.move_predictor = nn.Sequential(
            nn.Linear(d_model * 65, 2048),  # 64 squares + additional features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 4864),  # Total possible moves
        )

    def _create_square_mask(self, sz: int) -> torch.Tensor:
        """Create mask for valid squares."""
        mask = torch.triu(torch.ones(sz, sz)) == 1
        mask = mask.float().masked_fill(mask == 0, float("-inf"))
        return mask

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

        # Create mask if not provided
        if mask is None:
            mask = self._create_square_mask(65).to(board_state.device)

        # Single transformer pass
        output = self.transformer(encoder_input, mask)

        # Direct prediction from transformer output
        output_flat = output.reshape(batch_size, -1)
        move_logits = self.move_predictor(output_flat)

        return move_logits

    def parameter_count(self) -> int:
        """Returns the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
