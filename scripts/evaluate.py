from pathlib import Path

import chess
import click
import torch
from tqdm import tqdm

import wandb
from chess_ai.data.dataset import ChessPolicyDataset
from chess_ai.models.cnn.model import ChessAISmaller
from chess_ai.models.transformer.model import ChessTransformer
from chess_ai.training.trainer import calculate_metrics
from chess_ai.utils.chess_utils import decode_move_index


def load_model(
    checkpoint_path: Path, model_type: str, device: torch.device
) -> torch.nn.Module:
    """Load a model from a checkpoint file."""
    if model_type == "cnn":
        model = ChessAISmaller()
    else:
        model = ChessTransformer()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model.to(device)


def evaluate_model(
    model: torch.nn.Module,
    dataset: ChessPolicyDataset,
    batch_size: int,
    device: torch.device,
    use_wandb: bool = False,
) -> dict[str, float]:
    """Evaluate a model on a dataset.

    Args:
        model: The model to evaluate
        dataset: Evaluation dataset
        batch_size: Batch size
        device: Device to evaluate on
        use_wandb: Whether to log results to W&B

    Returns:
        Dictionary of metric names and values
    """
    model.eval()
    dataloader = dataset.create_dataloader(batch_size)

    # Print dataset encoding info
    print("\nDataset Info:")
    print(f"Total positions: {len(dataset)}")

    # Sample a few positions and their moves
    for i in range(3):
        state, const, move = dataset[i]
        print(f"\nPosition {i}:")
        print(f"Move encoding: {move}")
        move = decode_move_index(move.item(), chess.WHITE)
        print("Decoded:", move)
        print("Reencoded:", move and dataset.encoder.encode_move(move))

    # Lists to store all outputs and targets
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for states, consts, moves in tqdm(dataloader, desc="Evaluating"):
            states = states.to(device)
            consts = consts.to(device)
            moves = moves.to(device)

            outputs = model(states, consts)

            # Store batch results
            all_outputs.append(outputs.cpu())
            all_targets.append(moves.cpu())

    # Concatenate all batches
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate metrics on entire dataset
    metrics = calculate_metrics(all_outputs, all_targets)

    if use_wandb:
        wandb.log({"eval/" + k: v for k, v in metrics.items()})

    return metrics


@click.command()
@click.option(
    "--checkpoint-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to model checkpoint file (.pth)",
)
@click.option(
    "--model-type",
    type=click.Choice(["cnn", "transformer"]),
    default="cnn",
    help="Model architecture to use",
)
@click.option(
    "--data-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to evaluation data",
)
@click.option(
    "--rating-range",
    type=str,
    required=True,
    help='Rating range to evaluate on (e.g. "1600-2000")',
)
@click.option("--batch-size", type=int, default=64, help="Evaluation batch size")
@click.option(
    "--use-wandb",
    is_flag=True,
    help="Log results to Weights & Biases",
)
def main(
    checkpoint_path: Path,
    model_type: str,
    data_path: Path,
    rating_range: str,
    batch_size: int,
    use_wandb: bool,
):
    """Evaluate a trained chess model on a dataset."""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize wandb if requested
    if use_wandb:
        wandb.init(
            project="chess_ai",
            config={
                "model_type": model_type,
                "rating_range": rating_range,
                "batch_size": batch_size,
            },
        )

    # Load model
    print("Loading model...")
    model = load_model(checkpoint_path, model_type, device)
    print("Model loaded successfully!")

    # Load evaluation data
    print("Loading evaluation data...")
    dataset = ChessPolicyDataset.from_files(data_path, rating_range)
    print(f"Loaded {len(dataset)} positions")

    # Evaluate
    metrics = evaluate_model(model, dataset, batch_size, device, use_wandb)

    # Print results
    print("\nEvaluation Results:")
    print("-" * 30)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()
