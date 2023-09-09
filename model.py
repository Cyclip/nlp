from typing import Tuple, Union, Optional, List
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torchcrf import CRF

import mapping
import constants


class SingleClassCRF(CRF):
    """
    This is for the case where there is only one class, so the CRF layer is not needed.
    Just for optimization purposes.
    """
    def __init__(self, num_tags: int, batch_first: bool = False):
        super().__init__(num_tags, batch_first=batch_first)

    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Calculate the negative log-likelihood of the given sequence of tags.

        Args:
            emissions (torch.Tensor): The emission scores for each label for each token
            tags (torch.Tensor): The ground truth labels
            mask (torch.Tensor, optional): The mask for ignoring padding. Defaults to None.

        Returns:
            torch.Tensor: The negative log-likelihood of the given sequence of tags
        """
        if self.num_tags == 1:
            return emissions
        else:
            return super().forward(emissions, tags, mask=mask)


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

        assert self.num_entities == 2

        # Text classification head (consider dropout)
        self.classification_head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.num_classes),
        )

        # Named entity recognition head
        self.ner_projection = nn.Linear(self.bert.config.hidden_size, self.num_entities)

        # self.crf = CRF(self.num_entities)
        self.crf = SingleClassCRF(self.num_entities, batch_first=True)
    
    def cuda(self, device=None) -> None:
        """Move the model to GPU.

        Args:
            device (Optional[int], optional): GPU device id. Defaults to None.
        """
        self.bert.cuda(device)
        self.classification_head.cuda(device)
        self.ner_projection.cuda(device)
        self.crf.cuda(device)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
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
        # get entities and labels from the mapping
        entities = self.mapping.get_entities()
        labels = self.mapping.get_class_labels()

        print(f"Entities: {entities}")
        print(f"Labels: {labels}")

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
        ner_logits = self.ner_projection(outputs.last_hidden_state)

        print(f"pre-crf: {ner_logits.shape}")

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

    def tokenize(self, input_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize and preprocess the input text.

        Args:
            input_text (str): The input text

        Returns:
        - input_ids (torch.Tensor): The tokenized input
        - attention_mask (torch.Tensor): Attention mask (used for ignoring padding)
        """
        # Tokenize and preprocess the input text
        input_dict = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=constants.MAX_SEQ_LEN,
            return_tensors='pt'
        )

        input_ids = input_dict['input_ids']
        attention_mask = input_dict['attention_mask']

        return input_ids, attention_mask

    def text_classification(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform text classification on the input.

        Args:
            input_ids (torch.Tensor): The tokenized input
            attention_mask (torch.Tensor): Attention mask (used for ignoring padding)

        Returns:
            torch.Tensor: The classification probabilities
        """
        # Forward pass through the model
        classification_logits = self.classification_head(self.bert(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :])

        # apply softmax to get probabilities
        classification_probs = nn.Softmax(dim=1)(classification_logits)

        return classification_probs

    def ner_extraction(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[str]:
        """Extract named entities from the input.

        Args:
            input_ids (torch.Tensor): The tokenized input
            attention_mask (torch.Tensor): Attention mask (used for ignoring padding)

        Returns:
            List[str]: The entity labels
        """
        # Forward pass through the model for NER
        ner_logits = self.ner_projection(self.bert(input_ids, attention_mask=attention_mask).last_hidden_state)

        return ner_logits
    
    def decode_ner_logits(
        self,
        ner_logits: torch.Tensor,
        attention_mask: torch.Tensor,
        threshold: Optional[float] = None
    ) -> List[Tuple[str, Tuple[int, int]]]:
        """Decode the NER logits to entity labels and spans.

        Args:
            ner_logits (torch.Tensor): The NER logits
            attention_mask (torch.Tensor): The attention mask for ignoring padding
            threshold (Optional[float]): Optional threshold for filtering entities
                                         A common value is 0.5

        Returns:
            List[Tuple[str, Tuple[int, int]]]: The entity labels and their spans
        """
        # Apply the CRF layer to decode the entity labels
        decoded_sequences = self.crf.decode(ner_logits, mask=attention_mask.byte())

        # Get the first decoded sequence (assuming batch size 1)
        decoded_sequence = decoded_sequences[0]

        # Convert the decoded sequence to a list of entity labels using the mapping
        entity_labels = [self.mapping.get_entity_label(idx) for idx in decoded_sequence]

        if threshold is not None:
            # Filter entities based on threshold and store spans
            filtered_entities = []
            current_entity = None
            start_idx = None
            for idx, label in enumerate(entity_labels):
                if label != "O":
                    if current_entity is None:
                        current_entity = label
                        start_idx = idx
                    else:
                        current_entity += label
                else:
                    if current_entity is not None:
                        end_idx = idx - 1
                        entity_length = len(current_entity.split())
                        if entity_length >= threshold:
                            filtered_entities.append((current_entity, (start_idx, end_idx)))
                        current_entity = None
            if current_entity is not None:
                end_idx = len(entity_labels) - 1
                entity_length = len(current_entity.split())
                if entity_length >= threshold:
                    filtered_entities.append((current_entity, (start_idx, end_idx)))
            return filtered_entities
        else:
            # Return entity labels without spans
            return [(label, None) for label in entity_labels]


if __name__ == "__main__":
    mapping_path = constants.MAP_PATH
    input_text = "turn the temperature to 70 degrees"

    model = MTLModel(mapping_path)

    # Tokenize the input
    input_ids, attention_mask = model.tokenize(input_text)

    # Perform text classification
    classification_probs = model.text_classification(input_ids, attention_mask)
    print("Classification Probabilities:", classification_probs)

    # Perform NER
    ner_logits = model.ner_extraction(input_ids, attention_mask)
    print("NER Logits:", ner_logits)

    # Decode NER logits
    entity_labels = model.decode_ner_logits(ner_logits, attention_mask, threshold=0.5)
    print("Entity Labels:", entity_labels)