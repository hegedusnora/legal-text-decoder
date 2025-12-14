import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = "/app"
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_FILES = [
    "data/export_1.json",
    "data/export_2.json",
    "data/export_3.json",
    "data/export_4.json"
]

OUTPUT_DIR = os.path.join(BASE_DIR, "output")

GDRIVE_FILE_IDS = [
    "1VXDzhqIPNiDliaVAGBz7UWyyshRNB1cM",
    "1efd5v0tWXmeAkZTSXMtZlBcwQ74Jw7G-",
    "1zyHuMopfRR_exp9WJKfgRkrV-gPY8FNK",
    "1AxJL3jcTX_d9ZF3hZJaHFfYnK4IF8zF3"
]


CLASS_WEIGHTS_PATH = os.path.join(OUTPUT_DIR, "class_weights.pth")
PROCESSED_DATA_PATH = os.path.join(OUTPUT_DIR, "processed_data.pt")
VOCAB_PATH = os.path.join(OUTPUT_DIR, "vocab.pth")
BASELINE_MODEL_PATH = os.path.join(OUTPUT_DIR, "baseline_model.pth")

NUM_CLASSES = 5
NUMBER_OF_EPOCHS = 50
NUMBER_OF_EPOCHS_CV = 30
PAD_IDX = 0
UNK_IDX = 1

LABEL_MAP = {
    "1-Nagyon nehezen érthető": 0,
    "2-Nehezen érthető": 1,
    "3-Többé/kevésbé megértem": 2,
    "4-Érthető": 3,
    "5-Könnyen érthető": 4
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
