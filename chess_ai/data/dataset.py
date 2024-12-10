from __future__ import annotations

import logging
from pathlib import Path

import chess.pgn
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from chess_ai.utils.chess_utils import EloRange

from .preprocessing import GameEncoder, StandardEncoder

logger = logging.getLogger(__name__)


class ChessPolicyDataset(Dataset):
    """Dataset for chess games."""

    def __init__(
        self,
        states: Tensor,
        state_consts: Tensor,
        moves: Tensor,
        encoder: GameEncoder | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            states: Tensor of shape (N, 8, 8, 12) containing board states
            state_consts: Tensor of shape (N, 3) containing additional state info
            moves: Tensor of shape (N,) containing move indices
            encoder: Optional encoder to use for new games
        """
        self.states = states
        self.state_consts = state_consts
        self.moves = moves
        self.encoder = encoder or StandardEncoder()

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        state = self.states[idx].permute(2, 0, 1)  # Change to (12, 8, 8)
        state_const = self.state_consts[idx]
        move = self.moves[idx]
        return state, state_const, move

    @classmethod
    def from_files(
        cls, data_dir: Path, rating_range: str, encoder: GameEncoder | None = None
    ) -> ChessPolicyDataset:
        """Load dataset from preprocessed tensor files.

        Args:
            data_dir: Directory containing tensor files
            rating_range: Rating range identifier (e.g., "1600-2000")
            encoder: Optional encoder to use for new games

        Returns:
            ChessDataset instance
        """
        states = torch.load(data_dir / f"states_tensors_{rating_range}.pt")
        state_consts = torch.load(data_dir / f"states_consts_tensors_{rating_range}.pt")
        moves = torch.load(data_dir / f"moves_tensors_{rating_range}.pt")

        logger.info("Loaded dataset with %d positions", len(states))

        return cls(states, state_consts, moves, encoder)

    def create_dataloader(
        self, batch_size: int, shuffle: bool = True, num_workers: int = 0
    ) -> DataLoader:
        """Create a DataLoader for this dataset.

        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes

        Returns:
            DataLoader instance
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=True,
        )

    @classmethod
    def from_pgn(
        cls,
        pgn_file: Path,
        base_rating: int,
        encoder: GameEncoder | None = None,
    ) -> "ChessPolicyDataset":
        """Create dataset directly from PGN file.

        Args:
            pgn_file: Path to PGN file
            base_rating: Base rating for filtering
            encoder: Optional encoder to use

        Returns:
            New dataset instance
        """
        encoder = encoder or StandardEncoder()
        num_games = EloRange.get_num_games(base_rating)

        states = []
        state_consts = []
        moves = []

        with open(pgn_file) as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                # Check rating range
                white_elo = game.headers.get("WhiteElo", "")
                black_elo = game.headers.get("BlackElo", "")
                if not (white_elo.isdigit() and black_elo.isdigit()):
                    continue

                if not (
                    base_rating <= int(white_elo) <= base_rating + 400
                    and base_rating <= int(black_elo) <= base_rating + 400
                ):
                    continue

                # Process game
                board = game.board()
                for move in game.mainline_moves():
                    state, consts = encoder.encode_game_state(board)
                    move_idx = encoder.encode_move(move)

                    states.append(state)
                    state_consts.append(consts)
                    moves.append(move_idx)

                    board.push(move)

                if len(states) >= num_games:
                    break

        return cls(
            torch.stack(states), torch.stack(state_consts), torch.tensor(moves), encoder
        )


class ChessValueDataset(Dataset):
    """Dataset for chess games with white win labels."""

    def __init__(
        self,
        states: Tensor,
        state_consts: Tensor,
        outcomes: Tensor,
        encoder: GameEncoder | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            states: Tensor of shape (N, 8, 8, 12) containing board states
            state_consts: Tensor of shape (N, 3) containing additional state info
            outcomes: Tensor of shape (N,) containing binary win/loss labels
            encoder: Optional encoder to use for new games
        """
        self.states = states
        self.state_consts = state_consts
        self.outcomes = outcomes
        self.encoder = encoder or StandardEncoder()

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        state = self.states[idx].permute(2, 0, 1)  # Change to (12, 8, 8)
        state_const = self.state_consts[idx]
        outcome = self.outcomes[idx]
        return state, state_const, outcome

    @classmethod
    def from_files(
        cls, data_dir: Path, rating_range: str, encoder: GameEncoder | None = None
    ) -> ChessValueDataset:
        """Load dataset from preprocessed tensor files.

        Args:
            data_dir: Directory containing tensor files
            rating_range: Rating range identifier (e.g., "1600-2000")
            encoder: Optional encoder to use for new games

        Returns:
            ChessValueDataset instance
        """
        states = torch.load(data_dir / f"states_tensors_{rating_range}.pt")
        state_consts = torch.load(data_dir / f"states_consts_tensors_{rating_range}.pt")
        outcomes = torch.load(data_dir / f"outcomes_tensors_{rating_range}.pt")

        logger.info("Loaded dataset with %d positions", len(states))

        return cls(states, state_consts, outcomes, encoder)

    def create_dataloader(
        self, batch_size: int, shuffle: bool = True, num_workers: int = 0
    ) -> DataLoader:
        """Create a DataLoader for this dataset.

        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes

        Returns:
            DataLoader instance
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=True,
        )

    @classmethod
    def from_pgn(
        cls,
        pgn_file: Path,
        base_rating: int,
        encoder: GameEncoder | None = None,
    ) -> "ChessValueDataset":
        """Create dataset directly from PGN file.

        Args:
            pgn_file: Path to PGN file
            base_rating: Base rating for filtering
            encoder: Optional encoder to use

        Returns:
            New dataset instance
        """
        encoder = encoder or StandardEncoder()
        num_games = EloRange.get_num_games(base_rating)

        states = []
        state_consts = []
        outcomes = []

        with open(pgn_file) as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                white_elo = game.headers.get("WhiteElo", "")
                black_elo = game.headers.get("BlackElo", "")
                if not (white_elo.isdigit() and black_elo.isdigit()):
                    continue

                if not (
                    base_rating <= int(white_elo) <= base_rating + 400
                    and base_rating <= int(black_elo) <= base_rating + 400
                ):
                    continue

                result = game.headers.get("Result", "*")

                board = game.board()
                for move in game.mainline_moves():
                    state, consts = encoder.encode_game_state(board)

                    if result == "1-0":
                        outcome = 1
                    elif result == "0-1":
                        outcome = 0
                    else:
                        outcome = 0.5

                    states.append(state)
                    state_consts.append(consts)
                    outcomes.append(outcome)

                    board.push(move)

                if len(states) >= num_games:
                    break

        return cls(
            torch.stack(states),
            torch.stack(state_consts),
            torch.tensor(outcomes, dtype=torch.float32),
            encoder,
        )
