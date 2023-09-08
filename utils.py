from typing import Tuple
from model import MTLModel
from data import MTLDataset
from transformers import BertTokenizer
import torch
from torch import Tensor
import matplotlib.pyplot as plt

def _predict(
        model: MTLModel,
        input_ids: Tensor,
        attention_mask: Tensor,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
    """Predict the output given the input.

    Args:
        model (MTLModel): The multi-task learning model
        input_ids (torch.Tensor): The tokenized input
        attention_mask (torch.Tensor): Attention mask (used for ignoring padding)
        device (torch.device): The device to run the model on

    Returns:
        torch.Tensor: The predicted output
    """
    # Move data to the specified device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Set model to evaluation mode
    model.eval()

    # Deactivate autograd engine and move model to specified device
    with torch.no_grad():
        # Forward propagation
        classification_logits, ner_logits, loss = model(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))

        # Get the predicted output
        classification_output = torch.argmax(classification_logits, dim=1)
        ner_output = torch.argmax(ner_logits, dim=2)

    return classification_output, ner_output, loss



def predict(
        model: MTLModel,
        data: str,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[int, int, Tuple[int, int], float]:
    """Predict the output given the input.
    Converts the output to classification label idx, NER label idx + span, and loss.

    Args:
        model (MTLModel): The multi-task learning model
        data (str): The input data
        device (torch.device, optional): Device to run the model on. Defaults to torch.device("cpu").

    Returns:
    - classification_output (int): The predicted classification label index
    - ner_idx (int): The predicted NER label index
    - ner_span (Tuple[int, int]): The predicted NER span
    - loss (float): The loss of the model
    """
    # Tokenize the input
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenized_text = tokenizer.encode_plus(
        data,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Get the input_ids and attention_mask
    input_ids = tokenized_text["input_ids"].squeeze(0)
    attention_mask = tokenized_text["attention_mask"].squeeze(0)

    # Get the predicted output
    classification_output, ner_output, loss = _predict(model, input_ids, attention_mask, device)

    # Get the NER span
    ner_span = None
    for i in range(len(ner_output[0])):
        if ner_output[0][i] != 0:
            ner_span = (i, i + 1)
            break
    
    classification_output = classification_output.item()
    ner_output = ner_output[0][i].item()
    ner_span = ner_span

    return classification_output, ner_output, ner_span, loss


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