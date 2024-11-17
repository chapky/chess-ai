import logging
import subprocess
from pathlib import Path

import click

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--remove-original",
    is_flag=True,
    help="Remove the compressed file after decompression",
)
def decompress(input_path: Path, remove_original: bool):
    """Decompress a .zst file."""
    if not input_path.suffix == ".zst":
        raise click.BadParameter("Input file must have .zst extension")

    output_path = input_path.with_suffix("")

    if output_path.exists():
        logger.info("Decompressed file already exists at %s", output_path)
        return

    logger.info("Decompressing %s...", input_path)
    subprocess.run(
        ["pzstd", "-d", str(input_path)],
        check=True,
    )

    if remove_original:
        input_path.unlink()
        logger.info("Removed original compressed file")

    logger.info("Decompression complete: %s", output_path)


if __name__ == "__main__":
    decompress()

