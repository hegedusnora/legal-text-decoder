# Utility functions
# Common helper functions used across the project.
import logging
import sys
import  re
import torch
import torch.nn as nn
import config

def setup_logger(name=__name__):
    """
    Sets up a logger that outputs to the console (stdout).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def simple_tokenizer(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

def load_config():
    pass

def log_hyperparameters(logger):
    """Logs the hyperparameters from config.py."""
    logger.info("--- Configuration & Hyperparameters ---")
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Batch Size (Hardcoded in loaders): 8 or 16")
    logger.info(f"Number of Classes: {config.NUM_CLASSES}")
    logger.info(f"Vocab Path: {config.VOCAB_PATH}")
    logger.info("---------------------------------------")

def log_model_summary(logger, model):
    """Logs the model architecture and parameter count."""
    logger.info("--- Model Summary ---")
    logger.info(str(model))
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total Parameters: {total_params}")
    logger.info(f"Trainable Parameters: {trainable_params}")
    logger.info("---------------------")