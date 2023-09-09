from typing import Tuple
from model import MTLModel
from data import MTLDataset
import torch
from torch import Tensor
import matplotlib.pyplot as plt

def predict(
        input_text: str,
        model: MTLModel,
        threshold: float = 0.5,
    ) -> Tuple[int, Tensor, Tensor, Tensor]:
    """Perform text classification and named entity recognition on the input text.

    Args:
        input_text (str): The input text
        model (MTLModel): The model
        threshold (float, optional): The threshold for the NER logits. Defaults to 0.5.
        
    Returns:
        Tensor: The classification probabilities
        Tensor: The NER logits
        Tensor: The NER labels
    """
    # Use GPU if available
    if torch.cuda.is_available():
        model = model.cuda()

    # Tokenize the input
    input_ids, attention_mask = model.tokenize(input_text)

    # Perform text classification
    classification_probs = model.text_classification(input_ids, attention_mask)

    # Perform NER
    ner_logits = model.ner_extraction(input_ids, attention_mask)

    # Decode NER logits
    entity_labels = model.decode_ner_logits(ner_logits, attention_mask, threshold=threshold)

    return classification_probs, ner_logits, entity_labels


def plot_training_data(
        classification_loss: list,
        entity_loss: list,
        total_loss: list,
        title: str = "Training Loss",
    ):
    """Plot the training data.

    Args:
        classification_loss (list): The training loss history for text classification
        entity_loss (list): The training loss history for named entity recognition
        total_loss (list): The training loss history for the entire model
    """
    plt.title(title)
    plt.plot(classification_loss, label="Classification Loss", color="orange")
    plt.plot(entity_loss, label="Entity Loss", color="green")
    plt.plot(total_loss, label="Total Loss", color="blue")
    plt.legend()
    plt.show()