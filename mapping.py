from typing import Tuple, List, Optional, Any
import json
from constants import MAP_PATH

class MTLMapping:
    """
    Mapping object used to map numerical values to their corresponding labels, and vice versa.

    Example mapping:
    ```
    {
        "classification_mapping": {
            "id_to_label": {
                "0": "change_temperature",
                "1": "toggle_lights",
                "2": "lock_door"
            },
            "label_to_id": {
                "change_temperature": 0,
                "toggle_lights": 1,
                "lock_door": 2
            }
        },
        "entity_mapping": {
            "id_to_label": {
                "1": "number",
                "2": "boolean"
            },
            "label_to_id": {
                "number": 1,
                "boolean": 2
            }
        }
    }
    ```

    NOTE: Entity mapping is 1-indexed because 0 is used to denote no entity.

    Static attributes:
    - Entity
        str: The entity label
        Any: The entity value
    - Classification (str): The classification label
    """
    Entity = Optional[Tuple[str, Any]]
    Classification = Optional[str]


    def __init__(self, map_path: str):
        """Create an classification & entity mapping object.

        Args:
            map_path (str): Path to the mapping JSON file
        """
        self.map_path = map_path
        self.classification_map, self.entity_map = self._load_map()
        self.num_classes = len(self.classification_map["id_to_label"])
        self.num_entities = len(self.entity_map["id_to_label"])
    
    @classmethod
    def from_dict(cls, mapping: dict):
        """Create an classification & entity mapping object from a mapping dict.

        Args:
            mapping (dict): The mapping dict

        Returns:
            MTLMapping: The mapping object
        """
        # Get the classification and entity mapping
        classification_map = mapping["classification_mapping"]
        entity_map = mapping["entity_mapping"]

        # Create the mapping object
        mapping_obj = cls(MAP_PATH)
        mapping_obj.classification_map = classification_map
        mapping_obj.entity_map = entity_map
        mapping_obj.num_classes = len(classification_map["id_to_label"])
        mapping_obj.num_entities = len(entity_map["id_to_label"])

        return mapping_obj
    
    def _load_map(self) -> Tuple[dict, dict]:
        """Load the mapping from a JSON file.

        Returns:
            Tuple[dict, dict]: The classification and entity mapping
        """
        # Load the mapping
        mapping = json.load(open(self.map_path, "r"))

        # Get the classification and entity mapping
        classification_map = mapping["classification_mapping"]
        entity_map = mapping["entity_mapping"]

        return classification_map, entity_map

    def get_class_label(self, label: str) -> str:
        """Get the classification label given the numerical index.

        Args:
            label (str): The numerical index

        Returns:
            str: The classification label
        """
        return self.classification_map["id_to_label"][str(label)]

    def get_class_idx(self, label: str) -> int:
        """Get the classification index given the label.

        Args:
            label (str): The classification label

        Returns:
            int: The numerical index
        """
        return self.classification_map["label_to_id"][label]
    
    def get_entity_label(self, label: str) -> str:
        """Get the entity label given the numerical index.

        Args:
            label (str): The numerical index

        Returns:
            str: The entity label
        """
        return self.entity_map["id_to_label"][str(label)]
    
    def get_entity_idx(self, label: str) -> int:
        """Get the entity index given the label.

        Args:
            label (str): The entity label

        Returns:
            int: The numerical index
        """
        return self.entity_map["label_to_id"][label]