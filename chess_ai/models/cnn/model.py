from __future__ import annotations
from chess import Board, Color, Move
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from chess_ai.data.preprocessing import GameEncoder
from chess_ai.models.base import ChessPolicyModel, ChessValueModel, get_move
from chess_ai.utils.chess_utils import decode_move_index


class ChessAISmaller(nn.Module):
    """A smaller CNN architecture for chess move prediction.

    This class implements the ChessPolicyModel protocol through its
    method signatures rather than explicit inheritance.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.ln1 = nn.LayerNorm([64, 8, 8])
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.ln2 = nn.LayerNorm([128, 4, 4])
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.ln3 = nn.LayerNorm([256, 2, 2])
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256 + 3, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 4864)

    def forward(self, board_state: Tensor, additional_params: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            board_state: Tensor of shape (batch_size, 12, 8, 8)
                12 channels represent different piece types and colors
            additional_params: Tensor of shape (batch_size, 3)
                Contains [queen_castling_right, king_castling_right, is_white_turn]

        Returns:
            Tensor of shape (batch_size, 4864) representing move probabilities
        """
        x = F.relu(self.ln1(self.conv1(board_state)))
        x = F.max_pool2d(x, (2, 2))

        x = F.relu(self.ln2(self.conv2(x)))
        x = F.max_pool2d(x, (2, 2))

        x = F.relu(self.ln3(self.conv3(x)))
        x = self.global_avg_pool(x).view(-1, 256)
        x = torch.cat((x, additional_params), 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

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


class ChessAITiny(nn.Module):
    """A tiny CNN architecture for chess move prediction.

    This class implements the ChessPolicyModel protocol through its
    method signatures rather than explicit inheritance.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 36, kernel_size=3, padding=1)
        self.ln1 = nn.LayerNorm([36, 8, 8])
        self.conv2 = nn.Conv2d(36, 64, kernel_size=3, padding=1)
        self.ln2 = nn.LayerNorm([64, 4, 4])
        self.conv3 = nn.Conv2d(64, 96, kernel_size=3, padding=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(96 + 3, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 4864)

    def forward(self, board_state: Tensor, additional_params: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            board_state: Tensor of shape (batch_size, 12, 8, 8)
                12 channels represent different piece types and colors
            additional_params: Tensor of shape (batch_size, 3)
                Contains [queen_castling_right, king_castling_right, is_white_turn]

        Returns:
            Tensor of shape (batch_size, 4864) representing move probabilities
        """
        x = F.relu(self.ln1(self.conv1(board_state)))
        x = F.max_pool2d(x, (2, 2))

        x = F.relu(self.ln2(self.conv2(x)))
        x = F.max_pool2d(x, (2, 2))

        x = F.relu(self.ln3(self.conv3(x)))
        x = self.global_avg_pool(x).view(-1, 96)
        x = torch.cat((x, additional_params), 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

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


class ChessAIValue(nn.Module):
    """A CNN architecture for chess position evaluation.

    This class implements the ChessValueModel protocol through its
    method signatures rather than explicit inheritance.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.ln1 = nn.LayerNorm([64, 8, 8])
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.ln2 = nn.LayerNorm([128, 4, 4])
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.ln3 = nn.LayerNorm([256, 2, 2])
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256 + 3, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, board_state: Tensor, additional_params: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            board_state: Tensor of shape (batch_size, 12, 8, 8)
                12 channels represent different piece types and colors
            additional_params: Tensor of shape (batch_size, 3)
                Contains [queen_castling_right, king_castling_right, is_white_turn]

        Returns:
            Tensor of shape (batch_size, 1) representing position evaluation
        """
        x = F.relu(self.ln1(self.conv1(board_state)))
        x = F.max_pool2d(x, (2, 2))

        x = F.relu(self.ln2(self.conv2(x)))
        x = F.max_pool2d(x, (2, 2))

        x = F.relu(self.ln3(self.conv3(x)))
        x = self.global_avg_pool(x).view(-1, 256)
        x = torch.cat((x, additional_params), 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

    @staticmethod
    def initialize_from_smaller_model(smaller_model_path: str) -> ChessAIValue:
        smaller_model = ChessAISmaller()
        checkpoint = torch.load(smaller_model_path)
        if "model_state_dict" in checkpoint:
            smaller_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            smaller_model.load_state_dict(checkpoint)

        value_model = ChessAIValue()

        # Copy weights from the smaller model to the value model
        # This will copy weights for all layers except the last fully connected layer
        value_model.conv1.weight.data = smaller_model.conv1.weight.data
        value_model.conv1.bias.data = smaller_model.conv1.bias.data

        value_model.ln1.weight.data = smaller_model.ln1.weight.data
        value_model.ln1.bias.data = smaller_model.ln1.bias.data

        value_model.conv2.weight.data = smaller_model.conv2.weight.data
        value_model.conv2.bias.data = smaller_model.conv2.bias.data

        value_model.ln2.weight.data = smaller_model.ln2.weight.data
        value_model.ln2.bias.data = smaller_model.ln2.bias.data

        value_model.conv3.weight.data = smaller_model.conv3.weight.data
        value_model.conv3.bias.data = smaller_model.conv3.bias.data

        value_model.ln3.weight.data = smaller_model.ln3.weight.data
        value_model.ln3.bias.data = smaller_model.ln3.bias.data

        value_model.fc1.weight.data = smaller_model.fc1.weight.data
        value_model.fc1.bias.data = smaller_model.fc1.bias.data

        value_model.dropout.p = smaller_model.dropout.p

        # The fc2 layer will remain randomly initialized

        return value_model
