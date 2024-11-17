import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ChessAISmaller(nn.Module):
    """A smaller CNN architecture for chess move prediction.

    This class implements the ChessModel protocol through its
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
