import logging
from pathlib import Path

import click
import torch

from chess_ai.data.dataset import ChessPolicyDataset
from chess_ai.data.preprocessing import StandardEncoder
from chess_ai.utils.chess_utils import EloRange

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing PGN files",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default="./processed_data",
    help="Directory to save processed tensors",
)
@click.option(
    "--rating-ranges",
    type=str,
    default="1600,2000",
    help="Comma-separated list of rating ranges to process",
)
def preprocess(input_dir: Path, output_dir: Path, rating_ranges: str):
    """Preprocess chess games into tensor format."""
    output_dir.mkdir(exist_ok=True)

    # Parse rating ranges
    ranges = [int(r) for r in rating_ranges.split(",")]

    encoder = StandardEncoder()

    for base_rating in ranges:
        logger.info("Processing rating range %d-%d", base_rating, base_rating + 400)

        # Find newest PGN file
        pgn_files = sorted(input_dir.glob("**/*.pgn"))
        if not pgn_files:
            raise click.BadParameter(f"No PGN files found in {input_dir}")

        pgn_file = pgn_files[-1]

        # Create dataset
        dataset = ChessPolicyDataset.from_pgn(pgn_file, base_rating, encoder)

        # Save tensors using the exact format expected by train.py
        range_str = EloRange.get_range_str(base_rating)

        torch.save(dataset.states, output_dir / f"states_tensors_{range_str}.pt")
        torch.save(
            dataset.state_consts, output_dir / f"states_consts_tensors_{range_str}.pt"
        )
        torch.save(dataset.moves, output_dir / f"moves_tensors_{range_str}.pt")

        logger.info("Saved %d positions for range %s", len(dataset), range_str)


if __name__ == "__main__":
    preprocess()
