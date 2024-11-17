# Chess AI

A deep learning chess engine trained on Lichess game data. This project implements both CNN and Transformer architectures for chess move prediction, with a tiny UI system and training pipeline.

## Installation

1. Install Poetry (package manager):

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Clone the repository:

```bash
git clone https://github.com/yourusername/chess-ai.git
cd chess-ai
```

3. Install dependencies:

```bash
poetry install
```

## Project Structure

```
chess-ai/
├── chess_ai/
│   ├── models/
│   │   ├── cnn/          # CNN architecture
│   │   └── transformer/  # Transformer architecture
│   ├── data/             # Data loading and processing
│   ├── training/         # Training logic
│   ├── ui/               # User interfaces
│   └── utils/            # Helper functions
├── scripts/              # Command-line tools
└── notebooks/            # Jupyter notebooks
```

## Usage

### Training

Train a model using the command-line interface:

```bash
poetry run python scripts/train.py \
    --model-type cnn \
    --data-path /path/to/pgn/data \
    --rating-range "1600-2000" \
    --batch-size 64 \
    --epochs 10 \
    --learning-rate 0.001
```

Available options:

-   `--model-type`: Choose between 'cnn' or 'transformer'
-   `--rating-range`: ELO rating range for training data
-   `--batch-size`: Training batch size
-   `--epochs`: Number of training epochs
-   `--learning-rate`: Learning rate
-   `--save-dir`: Directory to save model checkpoints

### Playing Against the Model

Launch the Jupyter notebook interface:

```bash
poetry run jupyter notebook notebooks/play.ipynb
```

The notebook provides an interactive chessboard where you can play against the trained model.

## Model Architectures

TODO: Check and complete this section

### CNN Model

The CNN architecture uses:

-   Convolutional layers with batch normalization
-   Residual connections
-   Global average pooling
-   Dense layers for move prediction

Input format:

-   12-channel 8x8 board representation (6 piece types × 2 colors)
-   Additional features (castling rights, turn indicator)

### Transformer Model

The experimental transformer architecture includes:

-   Separate embeddings for pieces and positions
-   Multi-head self-attention
-   Positional encoding
-   Encoder-decoder architecture

This architecture is still under development and may not perform as well as the CNN model.

## Development

### Running Tests

The tests were made really quickly and sloppily, will not work in your environment most likely.

```bash
poetry run pytest
```
