import subprocess
from pathlib import Path

import click


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
    """Download chess game data from Lichess."""
    output_dir.mkdir(exist_ok=True)

    # Create month directory
    month_str = f"{month:02d}"
    month_dir = output_dir / f"{year}.{month_str}"
    month_dir.mkdir(exist_ok=True)

    # Download and decompress file
    filename = f"lichess_db_standard_rated_{year}-{month_str}"
    url = f"https://database.lichess.org/standard/{filename}.pgn.zst"

    print(f"Downloading {url}...")
    subprocess.run(
        ["curl", url, "--output", str(month_dir / f"{filename}.pgn.zst")],
        check=True,
    )

    print("Decompressing...")
    subprocess.run(
        ["pzstd", "-d", str(month_dir / f"{filename}.pgn.zst")],
        check=True,
    )

    # Cleanup
    (month_dir / f"{filename}.pgn.zst").unlink()

    print(f"Data downloaded and extracted to {month_dir}")


if __name__ == "__main__":
    download()
