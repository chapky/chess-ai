from __future__ import annotations

from chess_ai.models.cnn.model import ChessAISmaller
from chess_ai.models.transformer.model import ChessTransformer
from chess_ai.models.base import ChessPolicyModel

from torch import Tensor


class Node:
    def __init__(self, state: Tensor, parent: Node, eval_model: ChessPolicyModel, rollout_model: ChessPolicyModel):
        self.state = state
        self.parent = parent
        self.model = eval_model
        self.children: list[Node] = []
        self.evaluation = 0

    def rollout(self):
        pass


class MCTS(ChessPolicyModel):
    def __init__(self):
        self.model = ChessAISmaller()
        self._tree = None
