import json
import torch
import os
import gdown
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from torchtext.vocab import vocab
import config
from utils import setup_logger, simple_tokenizer

logger = setup_logger()

def download_data():
    os.makedirs(config.DATA_DIR, exist_ok=True)

    for file_id, out_file in zip(config.GDRIVE_FILE_IDS, config.RAW_DATA_FILES):
        if os.path.exists(out_file):
            logger.info(f"{out_file} already exists, skipping download")
            continue

        logger.info(f"Downloading raw data from Google Drive to {out_file}")
        gdown.download(id=file_id, output=out_file, quiet=False)

def analyze_data(all_raw):
    texts = []
    labels = []

    for raw in all_raw:
        for item in raw:
            if not item.get("annotations"):
                continue
            texts.append(item["data"]["text"])
            labels.append(item["annotations"][0]["result"][0]["value"]["choices"][0])

    lengths = [len(t.split()) for t in texts]

    logger.info("Data analysis started")
    logger.info(f"Number of samples: {len(texts)}")
    logger.info(f"Mean text length: {np.mean(lengths):.2f}")
    logger.info(f"Median text length: {np.median(lengths):.2f}")

    for l in sorted(set(labels)):
        logger.info(f"Label {l}: {labels.count(l)}")

def load_and_parse():
    logger.info("Loading raw JSON data")

    all_raw = []
    for path in config.RAW_DATA_FILES:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
            all_raw.append(raw)
            logger.info(f"Loaded {len(raw)} samples from {path}")

    analyze_data(all_raw)

    parsed = []
    for raw in all_raw:
        for item in raw:
            try:
                text = item["data"]["text"]
                label = config.LABEL_MAP[item["annotations"][0]["result"][0]["value"]["choices"][0]]
                parsed.append((text, label))
            except:
                continue

    logger.info(f"Parsed samples total: {len(parsed)}")
    return parsed

def build_vocab(texts):
    logger.info("Building vocabulary")
    counter = Counter()
    for t in texts:
        counter.update(simple_tokenizer(t))
    v = vocab(counter, specials=["<pad>", "<unk>"])
    v.set_default_index(v["<unk>"])
    logger.info(f"Vocabulary size: {len(v)}")
    return v

def preprocess():
    logger.info("Preprocessing pipeline started")

    download_data()
    data = load_and_parse()

    texts = [t for t, _ in data]
    labels = [l for _, l in data]

    v = build_vocab(texts)
    torch.save(v, config.VOCAB_PATH)
    logger.info("Vocabulary saved")

    encoded = []
    for t, l in zip(texts, labels):
        encoded.append(
            (l, torch.tensor([v[x] for x in simple_tokenizer(t)], dtype=torch.long))
        )

    train, test = train_test_split(
        encoded,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    counts = Counter(labels)
    total = sum(counts.values())
    weights = []

    for i in range(config.NUM_CLASSES):
        weights.append(total / (config.NUM_CLASSES * counts[i]))

    torch.save(torch.tensor(weights, dtype=torch.float), config.CLASS_WEIGHTS_PATH)
    logger.info(f"Class weights saved: {weights}")

    torch.save({"train": train, "test": test}, config.PROCESSED_DATA_PATH)

    logger.info(f"Train samples: {len(train)}")
    logger.info(f"Test samples: {len(test)}")
    logger.info("Preprocessing finished")

if __name__ == "__main__":
    preprocess()
