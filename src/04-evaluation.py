import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import config
from utils import setup_logger

logger = setup_logger()

class CVModel(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.embedding = nn.Embedding(vocab, 32, padding_idx=config.PAD_IDX)
        self.encoder = nn.GRU(32, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, config.NUM_CLASSES)

    def forward(self, x):
        x = self.embedding(x)
        _, h = self.encoder(x)
        h = torch.cat((h[-2], h[-1]), dim=1)
        return self.fc(h)

class BaselineModel(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.embedding = nn.Embedding(vocab, 64, padding_idx=config.PAD_IDX)
        self.encoder = nn.GRU(64, 128, batch_first=True)
        self.fc = nn.Linear(128, config.NUM_CLASSES)

    def forward(self, x):
        x = self.embedding(x)
        _, h = self.encoder(x)
        return self.fc(h.squeeze(0))

def collate(batch):
    labels, texts = zip(*batch)
    return torch.tensor(labels), pad_sequence(texts, batch_first=True, padding_value=config.PAD_IDX)

def save_confusion_matrix(cm, name):
    plt.figure()
    plt.imshow(cm, cmap='Blues')
    plt.title(name)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.savefig(os.path.join(config.OUTPUT_DIR, f"{name.lower()}_confusion_matrix.png"))
    plt.close()

def evaluate_model(name, model, loader):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for l, t in loader:
            t = t.to(config.DEVICE)
            out = model(t)
            p = torch.argmax(out, dim=1).cpu()
            preds.extend(p.tolist())
            labels.extend(l.tolist())

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    mae = mean_absolute_error(labels, preds)
    cm = confusion_matrix(labels, preds)

    logger.info(f"{name} | Accuracy: {acc:.3f}")
    logger.info(f"{name} | F1: {f1:.3f}")
    logger.info(f"{name} | MAE: {mae:.3f}")
    logger.info(f"{name} Confusion Matrix:\n{cm}")

    save_confusion_matrix(cm, name)

def evaluate_ensemble(models, loader):
    all_preds = []

    for model in models:
        model.eval()
        preds = []
        with torch.no_grad():
            for l, t in loader:
                t = t.to(config.DEVICE)
                out = torch.softmax(model(t), dim=1).cpu()
                preds.append(out)
        all_preds.append(torch.cat(preds, dim=0))

    final_preds = torch.stack(all_preds).mean(0)
    final_labels = torch.tensor([l for l, _ in loader.dataset])
    pred_classes = torch.argmax(final_preds, dim=1)

    acc = accuracy_score(final_labels, pred_classes)
    f1 = f1_score(final_labels, pred_classes, average="weighted")
    mae = mean_absolute_error(final_labels, pred_classes)
    cm = confusion_matrix(final_labels, pred_classes)

    logger.info(f"ENSEMBLE | Accuracy: {acc:.3f}")
    logger.info(f"ENSEMBLE | F1: {f1:.3f}")
    logger.info(f"ENSEMBLE | MAE: {mae:.3f}")
    logger.info(f"ENSEMBLE Confusion Matrix:\n{cm}")

    save_confusion_matrix(cm, "ENSEMBLE")

def evaluate():
    logger.info("Evaluation started")

    data = torch.load(config.PROCESSED_DATA_PATH)
    vocab = torch.load(config.VOCAB_PATH)
    loader = DataLoader(data["test"], batch_size=8, collate_fn=collate)

    baseline = BaselineModel(len(vocab)).to(config.DEVICE)
    baseline.load_state_dict(torch.load(config.BASELINE_MODEL_PATH))
    evaluate_model("BASELINE", baseline, loader)

    models = []
    for fold in range(5):
        model = CVModel(len(vocab)).to(config.DEVICE)
        model.load_state_dict(torch.load(os.path.join(config.OUTPUT_DIR, f"cv_model_fold_{fold}.pth")))
        models.append(model)

    evaluate_ensemble(models, loader)

    logger.info("Evaluation finished")

if __name__ == "__main__":
    evaluate()
