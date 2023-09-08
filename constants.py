import os

# The maximum sequence length the model can handle
MAX_SEQ_LEN = 512

# Training hyperparameters
LR = 1e-3
BATCH_SIZE = 32
EPOCHS = 10

# Default paths
TRAIN_PATH = "data/train.json"
TEST_PATH = "data/test.json"
MAP_PATH = "data/map.json"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pth")