import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import KFold
import numpy as np
import os
import config
from utils import setup_logger, log_hyperparameters, log_model_summary

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

def collate(batch):
    labels, texts = zip(*batch)
    return torch.tensor(labels), pad_sequence(texts, batch_first=True, padding_value=config.PAD_IDX)

def training():
    logger.info("K-Fold model development started")

    data = torch.load(config.PROCESSED_DATA_PATH)
    dataset = data["train"]
    vocab = torch.load(config.VOCAB_PATH)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for fold, (tr, va) in enumerate(kf.split(dataset)):
        logger.info(f"Fold {fold + 1} started")

        train_loader = DataLoader(Subset(dataset, tr), batch_size=8, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(Subset(dataset, va), batch_size=8, collate_fn=collate)

        model = CVModel(len(vocab)).to(config.DEVICE)
        if fold == 0:
            log_hyperparameters(logger)
            log_model_summary(logger, model)
        opt = optim.Adam(model.parameters(), lr=0.001)
        weights = torch.load(config.CLASS_WEIGHTS_PATH).to(config.DEVICE)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        best = 0

        for epoch in range(1, config.NUMBER_OF_EPOCHS_CV + 1):
            model.train()
            for l, t in train_loader:
                l, t = l.to(config.DEVICE), t.to(config.DEVICE)
                opt.zero_grad()
                loss = loss_fn(model(t), l)
                loss.backward()
                opt.step()

            model.eval()
            correct = total = 0
            with torch.no_grad():
                for l, t in val_loader:
                    l, t = l.to(config.DEVICE), t.to(config.DEVICE)
                    out = model(t)
                    p = torch.argmax(out, 1)
                    correct += (p == l).sum().item()
                    total += l.size(0)

            acc = correct / total
            logger.info(f"Fold {fold + 1} | Epoch {epoch} | Val acc: {acc:.3f}")

            if acc > best:
                best = acc
                torch.save(model.state_dict(), os.path.join(config.OUTPUT_DIR, f"cv_model_fold_{fold}.pth"))

        scores.append(best)
        logger.info(f"Fold {fold + 1} best acc: {best:.3f}")

    logger.info(f"Mean CV acc: {np.mean(scores):.3f}")
    logger.info(f"Std CV acc: {np.std(scores):.3f}")

if __name__ == "__main__":
    training()
