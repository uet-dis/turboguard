"""Default hyperparameters for TurboGuard.

These are framework defaults. Every value can be overridden via CLI
arguments at the dataset level — nothing is hardcoded to a specific dataset.
"""

# VQ-VAE architecture.
LATENT_DIM = 64  # Bottleneck dimension D.
NUM_EMBEDDINGS = 1024  # Codebook size K.
COMMITMENT_COST = 0.25  # Beta in the commitment loss (per van den Oord).
EMA_DECAY = 0.99  # Exponential Moving Average decay for codebook updates.
EMA_EPSILON = 1e-5  # Laplace smoothing in EMA to prevent div-by-zero.
VQVAE_EPOCHS = 20  # Default VQ-VAE training epochs.
DEAD_CODE_RESET_INTERVAL = 3  # Replace dead codes every N epochs.

# Training.
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
SEED = 42

# IsolationForest (hard-drop filter).
ISO_N_ESTIMATORS = 200  # Number of trees.
ISO_CONTAMINATION = 0.001  # Expected fraction of anomalies in benign data.
FPR_BUDGET = 1.5  # Maximum acceptable false positive rate (%).

# Calibration percentile sweep for IF threshold selection.
# Each percentile P means "block the bottom P% of benign IF scores".
# Lower P = more aggressive (blocks more, higher ADR but higher FPR).
# Higher P = more permissive (blocks less, lower FPR but lower ADR).
# The sweep finds the most aggressive P that stays within FPR_BUDGET.
CALIBRATION_PERCENTILES = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

# DNN grey-zone classifier.
DNN_HIDDEN_DIM = 128  # Base hidden dimension (first layer uses 2x).
DNN_MAX_EPOCHS = 50  # Maximum training epochs (early stopping applies).
DNN_PATIENCE = 10  # Early stopping patience (epochs without improvement).
