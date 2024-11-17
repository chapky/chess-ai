import logging
import subprocess
import sys
from pathlib import Path

import click

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("mount_point", type=click.Path())
def mount(input_file: str, mount_point: str):
    """Mount a .zst file using zstdmount."""
    input_path = Path(input_file)
    mount_path = Path(mount_point)

    if not input_path.suffix == ".zst":
        raise click.BadParameter("Input file must have .zst extension")

    # Create mount point if it doesn't exist
    mount_path.mkdir(parents=True, exist_ok=True)

    # Check if zstdmount is available
    try:
        subprocess.run(["which", "zstdmount"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        logger.error(
            "zstdmount not found. Please install zstdfuse (https://github.com/pjrinaldi/zstdfuse)."
        )
        sys.exit(1)

    logger.info("Mounting %s to %s...", input_path, mount_path)

    try:
        subprocess.run(["zstdmount", str(input_path), str(mount_path)], check=True)
        logger.info("Successfully mounted at %s", mount_path)
        logger.info("Use 'fusermount -u <mount_point>' to unmount when done")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to mount: {e}")
        sys.exit(1)


if __name__ == "__main__":
    mount()
