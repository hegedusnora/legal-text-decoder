import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import config
from utils import setup_logger, log_hyperparameters, log_model_summary

logger = setup_logger()

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

def train():
    logger.info("Baseline training started")

    data = torch.load(config.PROCESSED_DATA_PATH)
    vocab = torch.load(config.VOCAB_PATH)

    loader = DataLoader(data["train"], batch_size=16, shuffle=True, collate_fn=collate)

    model = BaselineModel(len(vocab)).to(config.DEVICE)
    log_hyperparameters(logger)
    log_model_summary(logger, model)
    opt = optim.Adam(model.parameters(), lr=0.001)
    weights = torch.load(config.CLASS_WEIGHTS_PATH).to(config.DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    best_loss = float("inf")
    patience = 3
    counter = 0

    for epoch in range(1, config.NUMBER_OF_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for l, t in loader:
            l, t = l.to(config.DEVICE), t.to(config.DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(t), l)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        logger.info(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
            torch.save(model.state_dict(), config.BASELINE_MODEL_PATH)
            logger.info("New best baseline model saved")
        else:
            counter += 1
            logger.info(f"No improvement for {counter} epoch(s)")

        if counter >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    logger.info("Baseline training finished")

if __name__ == "__main__":
    train()
