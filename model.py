from typing import Tuple, List, Union
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torchcrf import CRF

import mapping


class MTLModel(nn.Module):
    """
    This is a multi-task learning model for text classification and named entity recognition.
    It uses a pre-trained BERT model as the encoder, and has two heads for the two tasks.

    Named entity recognition head:
        NER head first uses a linear layer to project the BERT embeddings to a vector of size num_entities,
        then uses a CRF layer to decode the entity labels. This is because the entity labels are not independent
        of each other, and CRF can capture the dependencies between them.

        In order to extract the entities type and span, you need to implement a function to decode the CRF output.
        TODO: Implement a function to decode the CRF output
    
    Text classification head:
        Text classification head uses the first token's embedding from BERT as the representation of the sentence,
        then uses a linear layer to project it to a vector of size num_classes. To get the final predicted class,
        you need to apply a softmax function to the output and use a mapping from index to class label.

    CRFs are a specific type of machine learning model designed for sequence labeling tasks like NER.
    They are called "conditional" because they model the conditional probability of assigning labels to a sequence
    of words given the observed input data. In this case, the input data is the BERT embeddings of the input sequence.
    """

    def __init__(
        self,
        mapping_path: Union[str, dict],
        bert_model_name: str = "bert-base-uncased",
        hidden_size: int = 128,
    ):
        """Generate a multi-task learning model for text classification and named entity recognition.

        Args:
            mapping_path (Union[str, dict]): Path to the class/entity id-label mapping JSON file, or the mapping dict
            bert_model_name (str, optional): Pre-trained BERT model. Defaults to "bert-base-uncased".
        """
        super(MTLModel, self).__init__()

        # Load pre-trained BERT model and tokenizer
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        
        # Load the mapping
        if isinstance(mapping_path, str):
            self.mapping = mapping.MTLMapping(mapping_path)
        elif isinstance(mapping_path, dict):
            self.mapping = mapping.MTLMapping.from_dict(mapping_path)
        else:
            raise ValueError("Invalid mapping path or mapping dict.")
    
        self.num_classes = self.mapping.num_classes
        self.num_entities = self.mapping.num_entities

        # Text classification head (consider dropout)
        self.classification_head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.num_classes),
        )

        # Named entity recognition head
        # Make sure beyond identifying the type of entity,
        # you also extract and return the entity span (start and end index)
        self.crf = CRF(self.num_entities)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        entities: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward propagation of the model based on the input.

        Args:
            input_ids (torch.Tensor): The tokenized input
            attention_mask (torch.Tensor): Attention mask (used for ignoring padding)
            labels (torch.Tensor, optional): The labels for text classification. Defaults to None.
            entities (torch.Tensor, optional): The entities for named entity recognition. Defaults to None.

        Returns:
        - classification_logits (torch.Tensor): The logits for text classification
        - ner_logits (torch.Tensor): The logits for named entity recognition
        - total_loss (torch.Tensor): The total loss of the model
        """
        # Check input dimensions
        if input_ids.size(0) != attention_mask.size(0):
            raise ValueError("Input dimensions do not match.")

        # Encode input_ids and attention_mask using BERT
        outputs = self.bert(input_ids, attention_mask=attention_mask)

        # Text classification output
        # [:, 0, :] selects the first token's embeddings
        classification_logits = self.classification_head(outputs.last_hidden_state[:, 0, :])

        # Named entity recognition output
        # A linear layer here is possible, consider using it if needed
        ner_logits = outputs.last_hidden_state

        # Calculate loss for both tasks
        total_loss = 0

        if labels is not None:
            # Calculate classification loss
            classification_loss = nn.CrossEntropyLoss()(classification_logits, labels)
            total_loss += classification_loss
        if entities is not None:
            # Calculate NER loss
            ner_loss = -self.crf(ner_logits, entities, mask=attention_mask.byte(), reduction='mean')
            total_loss += ner_loss
        
        return classification_logits, ner_logits, total_loss