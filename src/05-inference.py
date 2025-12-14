import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import os
import config
from utils import setup_logger, simple_tokenizer

logger = setup_logger()

class InferenceModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_class):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=config.PAD_IDX)
        self.encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_class)

    def forward(self, text):
        x = self.embedding(text)
        _, hidden = self.encoder(x)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(hidden)

def predict(texts):
    vocab = torch.load(config.VOCAB_PATH)

    tokenized = []
    for text in texts:
        tokens = [vocab[token] for token in simple_tokenizer(text)]
        tokenized.append(torch.tensor(tokens, dtype=torch.long))

    batch = pad_sequence(tokenized, batch_first=True, padding_value=config.PAD_IDX).to(config.DEVICE)

    models = []
    for fold in range(5):
        model = InferenceModel(
            vocab_size=len(vocab),
            embed_dim=32,
            hidden_dim=64,
            num_class=config.NUM_CLASSES
        ).to(config.DEVICE)

        model_path = os.path.join(config.OUTPUT_DIR, f"cv_model_fold_{fold}.pth")
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        model.eval()
        models.append(model)

    with torch.no_grad():
        all_outputs = []
        for model in models:
            outputs = torch.softmax(model(batch), dim=1)
            all_outputs.append(outputs)
        
        avg_output = torch.stack(all_outputs).mean(0)
        preds = torch.argmax(avg_output, dim=1).cpu().tolist()

    for text, pred in zip(texts, preds):
        logger.info(f"Text: {text}")
        logger.info(f"Predicted readability level: {pred + 1}/5")

if __name__ == "__main__":
    samples = [
        "A Megrendelés leadása a Vásárló részéről ajánlattételnek minősül, amelynek az Eladóhoz történő megérkezésekor ajánlati kötöttséget eredményez a Vásárló oldalán. A Megrendelés leadása nem eredményezi az Eladó és Vásárló közötti szerződés létrejöttét. A Megrendelés leadása után a Vásárlóhoz e-mail, és SMS vagy Viber üzenet útján érkezett automatikus rendszerüzenet a Megrendelés rögzítéséről, illetve az ajánlat Eladóhoz történő beérkezéséről szól, így tájékoztató jellegű, és nem jelenti a Megrendelés Eladó általi visszaigazolását (első rendszerüzenet)",
        "Az Ügyfél/Vásárló ezen fizetési módot valamely vagy mindegyik bankkártya vonatkozásában bármikor törölheti, így inaktiválja (azaz megszünteti) a Payment by 1 click” (Fizetés 1 kattintással) fizetési módot a Saját fiókjában.",
        "Az MCOM értékhatárt és/vagy Telco értékhatárt a Szolgáltató akkor tekinti elértnek, ha az Előfizető részére az adott számlázási időszakon (továbbiakban: Periódus) belül kiállított számla végösszege, valamint az Előfizető által az adott Periódusban kezdeményezett díjfizetésre köteles forgalom még ki nem számlázott bruttó értéke eléri az MCOM értékhatárt és/vagy Telco értékhatárt."
    ]
    predict(samples)