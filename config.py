"""Central configuration for the AMC project.

All paths, hyperparameters, and class definitions live here.
Other modules import from this file so we have a single source of truth.
"""
from pathlib import Path

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_PKL = DATA_DIR / "RML2016.10a_dict.pkl"     # downloaded dataset
PROCESSED_DIR = DATA_DIR / "processed"           # cached numpy splits
RUNS_DIR = PROJECT_ROOT / "runs"                 # tensorboard / checkpoints
FIG_DIR = PROJECT_ROOT / "figures"

for d in (DATA_DIR, PROCESSED_DIR, RUNS_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------- Dataset ----------
# 11 classes, alphabetically sorted -> deterministic label index
MODULATIONS = [
    "8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK",
    "PAM4", "QAM16", "QAM64", "QPSK", "WBFM",
]
NUM_CLASSES = len(MODULATIONS)
MOD_TO_IDX = {m: i for i, m in enumerate(MODULATIONS)}
IDX_TO_MOD = {i: m for m, i in MOD_TO_IDX.items()}

SNR_RANGE = list(range(-20, 19, 2))   # -20, -18, ..., 18  (20 values)

# ---------- Splits ----------
SEED = 42
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# ---------- Training ----------
BATCH_SIZE = 256
NUM_EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.5
EARLY_STOP_PATIENCE = 10
NUM_WORKERS = 2

# ---------- Device ----------
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
