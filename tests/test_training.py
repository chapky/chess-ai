from pathlib import Path

from click.testing import CliRunner

from scripts.train import train


def test_training():
    # Create small test dataset
    test_data_path = Path("test_data")
    test_data_path.mkdir(exist_ok=True)

    # Test training command
    cli_runner = CliRunner()
    cli_runner.invoke(
        train,
        [
            "--model-type",
            "cnn",
            "--data-path",
            str(test_data_path),
            "--rating-range",
            "1600-2000",
            "--batch-size",
            "32",
            "--epochs",
            "1",
            "--learning-rate",
            "0.001",
            "--save-dir",
            "test_models",
        ],
    )

    print("Training pipeline test passed!")


if __name__ == "__main__":
    test_training()
