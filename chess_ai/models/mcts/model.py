from __future__ import annotations
from math import sqrt

from chess_ai.models.cnn.model import ChessAISmaller
from chess_ai.models.transformer.model import ChessTransformer
from chess_ai.models.base import ChessPolicyModel, ChessValueModel

from chess_ai.ui.base import GameController

from chess_ai.data.preprocessing import StandardEncoder
from chess_ai.utils.chess_utils import decode_move_index

import torch

from chess import Board, Move, pgn


state1 = []
state2 = []
consts1 = []
consts2 = []
probabilities1 = []
probabilities2 = []


class Node:
    def __init__(
        self,
        game_state: Board,
        parent: Node | None,
        policy_model: ChessPolicyModel,
        value_model: ChessValueModel,
        rollout_model: ChessPolicyModel,
        device: torch.device,
        move_to_it: Move | None = None,
        prior_probability: float = 1,
        prefer_rollout_coefficient: float = 0.5,
        exploration_coefficient: float = 5,
        verbose_level: int = 0,
    ):
        self.game_state = game_state
        self.parent = parent
        self.move_to_it = move_to_it

        self.prefer_rollout_coefficient = prefer_rollout_coefficient
        self.exploration_coefficient = exploration_coefficient

        self.policy_model = policy_model
        self.value_model = value_model
        self.rollout_model = rollout_model
        self.device = device

        self.prior_probability = prior_probability
        self.verbose_level = verbose_level

        self.children: list[Node] = []
        self.evaluation = 0
        self.visit_count = 0

        self.encoder = StandardEncoder()
        self.controller = GameController(
            None, rollout_model, self.encoder, device=device
        )

    def rollout_leaf(self) -> int:
        game = pgn.Game.from_board(self.game_state)
        while True:
            if self.verbose_level >= 2:
                print(game.board())
                print()
            try:
                move = self.controller._get_model_move(game.board())
            except StopIteration:
                print("Winner: None")
                return 0
            result = self.controller._make_move(game, move)
            if not result.game_over:
                game = game.next()
                continue
            if self.verbose_level >= 1:
                print("Winner: ", result.winner)
            if result.winner == "white":
                return 1
            if result.winner == "black":
                return -1
            return 0

    def value(self) -> float:
        # self.value_model.eval() # todo enable when this is replaced with an actual model
        with torch.no_grad():
            state, consts = self.encoder.encode_game_state(self.game_state)
            state = state.permute(2, 0, 1).unsqueeze(0).to(self.device)
            consts = consts.unsqueeze(0).to(self.device)
            result = self.value_model(state, consts)
            if self.verbose_level >= 1:
                print("value: ", result)
            return result[0]

    def evaluate_leaf(self):
        self.evaluation = (
            1 - self.prefer_rollout_coefficient
        ) * self.value() + self.prefer_rollout_coefficient * self.rollout_leaf()

    def get_children(self):
        self.policy_model.eval()
        with torch.no_grad():
            if self.verbose_level >= 2:
                print("Encoding game state...")
                print("Doing on game state: ", self.game_state)
            state, consts = self.encoder.encode_game_state(self.game_state)
            state = state.permute(2, 0, 1).unsqueeze(0).to(self.device)
            consts = consts.unsqueeze(0).to(self.device)
            global state1
            state1 = state
            global consts1
            consts1 = consts

            if self.verbose_level >= 2:
                print("Using the model to get children...")
            outputs = self.policy_model(state, consts)
            if self.verbose_level >= 2:
                print("Softmaxing...")
            probabilities = torch.softmax(outputs, dim=1)
            # if self.verbose_level >= 2: # WHYY DIFFERENT
            # print([i.item() for i in probabilities[0]])
            global probabilities1
            probabilities1 = probabilities
            if self.verbose_level >= 2:
                print("Filtering legal moves...")
            moves: list[tuple[Move, float]] = []
            for idx in range(len(probabilities[0])):
                try:
                    move = decode_move_index(int(idx), self.game_state.turn)
                except ValueError:
                    continue
                if move in self.game_state.legal_moves:
                    moves.append((move, probabilities[0][idx].item()))
            if self.verbose_level >= 2:
                print("Creating children...")
            for move in moves:
                board = self.game_state.copy()
                board.push(move[0])

                child = Node(
                    board,
                    self,
                    self.policy_model,
                    self.value_model,
                    self.rollout_model,
                    self.device,
                    move_to_it=move[0],
                    prior_probability=move[1],
                    prefer_rollout_coefficient=self.prefer_rollout_coefficient,
                    exploration_coefficient=self.exploration_coefficient,
                    verbose_level=self.verbose_level,
                )
                self.children.append(child)

    def select_child(self):
        if len(self.children) == 0:
            if self.verbose_level >= 2:
                print("Getting children...")
            self.get_children()
        if self.verbose_level >= 2:
            print("Getting bonuses...")
        bonuses = (
            self.exploration_coefficient
            * child.prior_probability
            * sqrt(self.visit_count)
            / (1 + child.visit_count)
            for child in self.children
        )
        if self.verbose_level >= 2:
            print("Getting choices...")
        if self.verbose_level >= 2:
            print("Getting max choice...")
        if self.verbose_level >= 1:
            choices = (
                (i, child.evaluation + bonus)
                for i, (child, bonus) in enumerate(zip(self.children, bonuses))
            )
            choices = sorted(choices, key=lambda x: x[1], reverse=True)
            for i, choice in choices:
                print(f"{choice}: \n{self.children[i].game_state}")
            max_index = choices[0][0]
        else:
            choices = [
                child.evaluation + bonus for child, bonus in zip(self.children, bonuses)
            ]
            max_choice = -float("inf")
            max_index = 0
            for i, choice in enumerate(choices):
                if choice > max_choice:
                    max_choice = choice
                    max_index = i
        return self.children[max_index]

    def expand(
        self,
    ):  # TODO: expand only when have been here for a few times. otherwise rollout again
        self.visit_count += 1
        if self.verbose_level >= 2:
            print("Selecting child...")
        child = self.select_child()
        if child.visit_count == 0:
            if self.verbose_level >= 2:
                print("Evaluating leaf...")
            child.evaluate_leaf()
            child.visit_count = 1
            parent = self
            if self.verbose_level >= 2:
                print("Propagating evaluation...")
            while parent:
                parent.evaluation = (
                    parent.evaluation * (parent.visit_count - 1) + child.evaluation
                ) / parent.visit_count
                parent = parent.parent
            return
        if self.verbose_level >= 2:
            print("Expanding child...")
        child.expand()

    def __eq__(self, other: Node | Board | object) -> bool:
        if isinstance(other, Board):
            return self.game_state == other
        if isinstance(other, Node):
            return self.game_state == other.game_state
        return False


class MCTS:
    def __init__(self, device: torch.device, verbose_level: int = 0):
        self.policy_model = ChessAISmaller().to(device)
        self.rollout_model = ChessAISmaller().to(device)
        self.value_model = lambda _, __: [0]
        new_game = pgn.Game()

        # WHYYY?
        state, consts = StandardEncoder().encode_game_state(new_game.board())
        state = state.permute(2, 0, 1).unsqueeze(0).to(device)
        consts = consts.unsqueeze(0).to(device)
        global state2
        state2 = state
        global consts2
        consts2 = consts
        # if verbose_level >= 2: # WHYY
        self.policy_model.eval()
        print("Doing on game state: ", new_game.board())
        with torch.no_grad():
            probabilities = self.policy_model(state, consts)
            global probabilities2
            probabilities2 = probabilities
        # print([i.item() for i in probabilities[0]])

        self.tree: Node = Node(
            new_game.board(),
            None,
            self.policy_model,
            self.value_model,
            self.rollout_model,
            device,
            prefer_rollout_coefficient=0.5,
            exploration_coefficient=5,
            verbose_level=verbose_level,
        )

    def change_board(self, board: Board):
        for i in self.tree.children:
            if i == board:
                self.tree = i

    def get_move(self, max_steps: int = 1) -> Move | None:
        for _ in range(max_steps):
            self.tree.expand()
        best_child = max(self.tree.children, key=lambda x: x.visit_count)
        return best_child.move_to_it


def print_tree(node: Node, current_depth: int = 0):
    board = str(node.game_state)
    tab = "   " * current_depth
    for i in board.split("\n"):
        print(tab + i)
    print(f"{tab}Visit count: {node.visit_count}, Evaluation: {node.evaluation}")
    for child in node.children:
        if not child.visit_count:
            continue
        print_tree(child, current_depth + 1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcts = MCTS(device, verbose_level=0)
    print_tree(mcts.tree)
    move = mcts.get_move(1)
    different = False
    for i in range(len(state1)):
        for j in range(len(state1[0])):
            for k in range(len(state1[0][0])):
                if state1[i][j][k][0] != state2[i][j][k][0]:
                    different = True
    if different:
        print("Different states")
        print(state1)
        print(state2)
    different = False
    for i in range(len(consts1)):
        if consts1[i][0] != consts2[i][0]:
            different = True
    if different:
        print("Different consts")
        print(consts1)
        print(consts2)
    different = False
    for i in range(len(probabilities1)):
        for j in range(len(probabilities1[0])):
            if probabilities1[i][j] != probabilities2[i][j]:
                different = True
    if different:
        print("Different probabilities")
        print(probabilities1)
        print(probabilities2)
    print(move)
    # print_tree(mcts.tree)
