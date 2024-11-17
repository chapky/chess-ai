import logging
import subprocess
from pathlib import Path

import click

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default="./data",
    help="Directory to store downloaded data",
)
@click.option(
    "--year",
    type=int,
    default=2020,
    help="Year of Lichess database",
)
@click.option(
    "--month",
    type=int,
    default=11,
    help="Month of Lichess database",
)
def download(output_dir: Path, year: int, month: int):
    """Download chess game data from Lichess if not already present."""
    output_dir.mkdir(exist_ok=True)

    # Create month directory
    month_str = f"{month:02d}"
    month_dir = output_dir / f"{year}.{month_str}"
    month_dir.mkdir(exist_ok=True)

    # Setup file paths
    filename = f"lichess_db_standard_rated_{year}-{month_str}"
    compressed_file = month_dir / f"{filename}.pgn.zst"

    if compressed_file.exists():
        logger.info(f"Compressed file already exists at {compressed_file}")
        return

    # Download file
    url = f"https://database.lichess.org/standard/{filename}.pgn.zst"
    logger.info(f"Downloading {url}...")

    subprocess.run(
        ["curl", url, "--output", str(compressed_file)],
        check=True,
    )

    logger.info(f"Download complete: {compressed_file}")


if __name__ == "__main__":
    download()
