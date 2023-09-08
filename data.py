from typing import Tuple, Any, Optional
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
import torch
from torch import Tensor
import pandas as pd

from constants import MAX_SEQ_LEN, TRAIN_PATH, TEST_PATH

class MTLDataset(Dataset):
    """
    Dataset used to finetune the multi-task learning model to perform text classification
    and named entity recognition.

    A dataset (json) should follow the format below:
        Column 1: text (str) - The input text
        Column 2: classification_label (int) - The label for text classification
        Column 3: entity_label (str) - The label for named entity recognition
    
    An example of a dataset is shown below:
    [
        {
            "text": "please turn the temperature down to 20 degrees",
            "classification_label": 0,
            "entity_label": [0, 0, 0, 0, 0, 0, 1, 0]
        },
        {
            "text": "its too bright, turn off the lights",
            "classification_label": 1,
            "entity_label": [0, 0, 0, 0, 0, 2, 0, 0]
        },
        {
            "text": "lock the door",
            "classification_label": 2,
            "entity_label": [2, 0, 0, 0, 0, 0, 0, 0]
        }
    ]
    
    Numerical values for `classification_label` and `entity_label` are mapped to their corresponding
    labels in `label_map` and `entity_map` respectively. This would be stored in `self.data` as a
    Pandas DataFrame.
    """
    def __init__(
        self,
        path: str,
        tokenizer: BertTokenizer,
        max_seq_len: int = MAX_SEQ_LEN,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data = self._load_data(path)

    def _load_data(self, path: str) -> Any:
        """Load the data from a JSON file.

        Args:
            path (str): The path to the JSON file

        Returns:
            Any: The data
        """
        # Load the data
        data = pd.read_json(path)
        
        # We do not necessarily need to convert the numerical labels to their corresponding
        # labels in `label_map` and `entity_map` here. We can do it later when we need to
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Get the data at the specified index.

        Args:
            idx (int): The index of the data

        Returns:
        - input_ids (torch.Tensor): The tokenized input
        - attention_mask (torch.Tensor): Attention mask (used for ignoring padding)
        - classification_label (torch.Tensor): The label for text classification
        - entity_label (torch.Tensor): The labels for named entity recognition
        """
        row = self.data.iloc[idx]

        # Tokenize the input text and encode the labels
        text = row["text"]
        classification_label = row["classification_label"]
        entity_label = row["entity_label"]
        tokenized_text = self.tokenizer.encode_plus(
            text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Convert the encoded input to PyTorch tensors
        input_ids = tokenized_text["input_ids"].squeeze(0)
        attention_mask = tokenized_text["attention_mask"].squeeze(0)
        classification_label = torch.tensor(classification_label)
        entity_label = torch.tensor(entity_label)

        return input_ids, attention_mask, classification_label, entity_label
    
    def collate_fn(
        self,
        batch: Tuple[Tensor, Tensor, Tensor, Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Collate the data into batches.

        Args:
            batch (Tuple[Tensor, Tensor, Tensor, Tensor]): The data in a batch

        Returns:
        - input_ids (torch.Tensor): The tokenized input
        - attention_mask (torch.Tensor): Attention mask (used for ignoring padding)
        - classification_label (torch.Tensor): The label for text classification
        - entity_label (torch.Tensor): The labels for named entity recognition
        """
        input_ids, attention_mask, classification_label, entity_label = zip(*batch)
        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        classification_label = torch.stack(classification_label)
        entity_label = torch.stack(entity_label)

        return input_ids, attention_mask, classification_label, entity_label
    
    @staticmethod
    def get_loader(
        path: str,
        tokenizer: BertTokenizer,
        max_seq_len: int = MAX_SEQ_LEN,
        batch_size: int = 32,
        shuffle: bool = False,
    ) -> DataLoader:
        """Get the data loader.

        Args:
            path (str): The path to the dataset
            tokenizer (BertTokenizer): The tokenizer
            max_seq_len (int, optional): The maximum sequence length. Defaults to MAX_SEQ_LEN.
            batch_size (int, optional): The batch size. Defaults to 32.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to False.

        Returns:
            DataLoader: The data loader
        """
        dataset = MTLDataset(path, tokenizer, max_seq_len)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=dataset.collate_fn,
        )

        return loader


if __name__ == "__main__":
    # Try to load the training data
    from constants import TRAIN_PATH

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dl = MTLDataset.get_loader(TRAIN_PATH, tokenizer, MAX_SEQ_LEN, shuffle=True)

    for batch in train_dl:
        input_ids, attention_mask, classification_labels, entity_labels = batch
        print(f"input_ids: {input_ids.shape}")
        print(f"attention_mask: {attention_mask.shape}")
        print(f"classification_labels: {classification_labels.shape}")
        print(f"entity_labels: {entity_labels.shape}")
        break
