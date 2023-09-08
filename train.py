from typing import Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

from data import MTLDataset
from model import MTLModel

from constants import (
    BATCH_SIZE,
    EPOCHS,
    LR,
    MAX_SEQ_LEN,
    MODEL_PATH,
    MODEL_DIR,
    TRAIN_PATH,
    TEST_PATH,
)

def train_model(
        model: MTLModel,
        save_to: str = None,
        classification_weight: float = 1.0,
        entity_weight: float = 1.0,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        lr: float = LR,
    ) -> Tuple[MTLDataset, List[float], List[float], List[float]]:
    """Train the model on a CSV dataset.

    Args:
        model (MTLModel): The multi-task learning model
        save_to (str, optional): File path to save the final model in. Defaults to None.
        classification_weight (float, optional): Weight for classification loss. Defaults to 1.0.
        entity_weight (float, optional): Weight for entity loss. Defaults to 1.0.

    Returns:
        - model (MTLModel): The trained model
        - classification_loss (List[float]): The training loss history for text classification
        - entity_loss (List[float]): The training loss history for named entity recognition
        - total_loss (List[float]): The training loss history for the entire model
    """

    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Make dataloader
    train_dl = MTLDataset.get_loader(TRAIN_PATH, tokenizer, MAX_SEQ_LEN, batch_size=batch_size, shuffle=True)

    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # metrics
    classification_loss = []
    entity_loss = []
    total_loss = []

    for epoch in range(epochs):
        model.train()

        total_classification_loss = 0.0
        total_entity_loss = 0.0
        total_loss = 0.0

        for batch in tqdm(train_dl, desc=f"Epoch {epoch + 1}"):
            input_ids, attention_mask, classification_labels, entity_labels = batch

            # forward propagation
            classification_logits, entity_logits, loss = model(
                input_ids, attention_mask, classification_labels, entity_labels
            )

            # calculate task-specific losses
            classification_loss = loss * classification_weight
            entity_loss = loss * entity_weight

            # backpropagation
            optimizer.zero_grad()  # dont accumulate gradients
            classification_loss.backward(retain_graph=True)   # retain_graph=True to prevent clearing the graph
            entity_loss.backward()
            optimizer.step()   # update parameters

            # update metrics
            total_classification_loss += classification_loss.item()
            total_entity_loss += entity_loss.item()
            total_loss += loss.item()

        # add to metrics
        classification_loss.append(total_classification_loss / len(train_dl))
        entity_loss.append(total_entity_loss / len(train_dl))
        total_loss.append(total_loss / len(train_dl))

    # save model
    if save_to:
        torch.save(model.state_dict(), save_to)
    
    return model, classification_loss, entity_loss, total_loss