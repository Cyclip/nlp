from typing import *
import numpy as np
from torch import Tensor

def viterbi_decode(crf_scores: Tensor, label_map: Dict[int, str], default_label: str = "O") -> List[Tuple[str, Tuple[int, int]]]:
    num_labels = crf_scores.size(1)
    seq_length = crf_scores.size(0)
    
    # Initialize the Viterbi scores and backtracking matrix
    viterbi = np.zeros((seq_length, num_labels))
    backpointer = np.zeros((seq_length, num_labels), dtype=int)
    
    # Initialize the scores for the first step
    viterbi[0] = crf_scores[0].detach().numpy()
    
    # Iterate through the sequence
    for t in range(1, seq_length):
        for label in range(num_labels):
            # Calculate the scores for transitioning from the previous label to the current label
            transition_scores = viterbi[t - 1] + crf_scores[t][label].detach().numpy()
            # Find the label with the highest score
            best_prev_label = np.argmax(transition_scores)
            viterbi[t][label] = transition_scores[best_prev_label]
            backpointer[t][label] = best_prev_label
    
    # Backtrack to find the best sequence of labels
    best_sequence = [np.argmax(viterbi[-1])]
    for t in range(seq_length - 1, 0, -1):
        best_prev_label = backpointer[t][best_sequence[-1]]
        best_sequence.append(best_prev_label)
    
    # Reverse the sequence and map labels to entity labels
    best_sequence.reverse()
    entity_spans = []
    start = 0
    current_label = None
    
    for t, label_idx in enumerate(best_sequence):
        if label_idx in label_map:
            label = label_map[label_idx]
        else:
            label = default_label
        if label.startswith("B-"):
            if current_label is not None:
                entity_spans.append((current_label, (start, t - 1)))
            current_label = label[2:]
            start = t
        elif label == "O":
            if current_label is not None:
                entity_spans.append((current_label, (start, t - 1)))
            current_label = None
    
    # Check for the last entity
    if current_label is not None:
        entity_spans.append((current_label, (start, seq_length - 1)))
    
    return entity_spans

crf_output = Tensor([[205, 505, 352,   7, 630, 167, 241, 262, 555, 262, 262, 505, 473, 262,
         505, 505, 505, 505, 262, 505, 505, 657, 262, 262, 505, 473, 505, 505,
         473, 731, 571, 276, 167, 167, 262, 262, 262, 657, 262, 505, 262, 505,
         262, 505, 324, 324, 505, 262, 262, 262, 505, 262, 505, 505, 473, 473,
         202, 505, 562, 571, 571, 262, 262, 262, 262, 262, 262, 262, 505, 505,
         505, 505, 262, 324, 505, 262, 262, 262, 262, 473, 473, 262, 505, 505,
         473, 473, 167, 562, 571, 571, 324, 262, 657, 262, 473, 262, 473, 262,
         262, 473, 473, 262, 324, 262, 505, 262, 262, 262, 505, 505, 473, 505,
         473, 473, 562, 571, 324, 262, 324, 262, 262, 262, 262, 262, 505, 505,
         505, 505,   2, 505, 505, 262, 505, 505, 571, 505, 262, 262, 505, 505,
         262, 262, 505, 505, 473, 473, 473, 505, 505, 262, 731, 505, 346, 657,
         262, 167, 167, 346, 346, 505, 505, 505, 473, 167, 262, 505, 167, 657,
         505, 505, 505, 473, 505, 473, 473, 473, 505, 505, 562, 638, 571, 167,
         657, 262, 262, 262, 657, 657, 262, 505, 505, 473, 505, 505, 505, 473,
           2, 324, 505, 262, 262, 657, 262, 657, 262, 505, 262, 657, 505, 505,
         505, 505, 262, 262, 324, 262, 505, 657, 657, 473, 262, 262, 505, 473,
         505, 262, 324,   2, 324, 505, 657, 657, 262, 167, 262, 262, 167, 262,
         505, 505, 505, 505, 473, 473, 505, 505, 505, 505, 657, 505, 505, 505,
         505, 505, 473, 473,   2, 276, 262, 657, 262, 657, 657, 167, 167, 657,
         473, 262, 473, 473, 473, 473, 324, 324, 505, 262, 167, 657, 262, 262,
         657, 657, 167, 505, 505, 473, 473, 473, 473, 473, 473, 473, 473, 167,
         324, 505, 262, 167, 657, 657, 262, 262, 262, 657, 262, 262, 262, 505,
         505, 505, 505, 505, 473, 505, 505, 731, 505, 657, 657, 167, 167, 262,
         473, 167, 657, 473, 505, 505, 473, 167, 262, 505, 505, 167, 657, 657,
         167, 505, 167, 346, 167, 505, 473, 505, 167, 473, 473, 505, 571, 324,
         262, 657, 262, 167, 262, 167, 167, 262, 262, 657, 262, 262, 505, 505,
         473, 473, 505, 505, 505, 473, 505, 505, 473,   2, 571, 505, 657, 262,
         505, 262, 262, 473, 473, 505, 505, 505, 473,   2, 473, 202, 505, 435,
         571, 505, 657, 262, 505, 571, 211, 657, 657, 167, 262, 262, 473, 473,
         473, 473, 473, 505, 505, 505, 505, 505, 505, 657, 505, 505, 262, 505,
         505, 505, 505, 262, 262, 276, 262, 657, 657, 167, 262, 473, 262, 473,
         167, 262, 505, 262, 262, 657, 262, 505, 505, 262, 505, 505, 262, 262,
         657, 167, 505, 505, 346, 505, 473, 505, 473, 505, 202, 132, 202, 571,
         609, 657, 167, 505, 657, 657, 262, 167, 262, 167, 262, 473, 473, 262,
         262, 262, 262, 262, 262, 657, 167, 346, 505, 505, 505, 505, 505, 473,
         505, 505, 167, 262, 167, 262, 505, 262, 262, 167, 657, 262, 505, 505,
         473, 167, 505, 505, 505, 505, 167, 505]])

label_map = {
    0: "O",
    1: "number",
    2: "boolean",
}
decoded_spans = viterbi_decode(crf_output, label_map)

print(decoded_spans)