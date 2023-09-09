import sys
import argparse
from constants import MAP_PATH, EPOCHS, BATCH_SIZE, LR
import warnings

def main() -> int:
    """
    Main function for argument parsing and running the model

    Optional:
        mapping (str): Path to the mapping file (default: data/map.json)

    Action (mutually exclusive):
    1.  --train (bool): Train the model
            --save (str): Save the model to the given path if training
            --plot (bool): Plot the training data
            --epochs (int): Number of epochs to train for (default: 10)
            --batchSize (int): Batch size (default: 32)
            --lr (float): Learning rate (default: 1e-3)

    2.  --test (str): Test the model on the given path, if not training

    3.  --text (str): Predict the output for the given text, if not training
            --path (str): Path to the model to load
            --rawData (bool): Print raw output from the model (default: False)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", help="path to the mapping file", default=MAP_PATH)
    
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--train", action="store_true", help="train the model")
    action.add_argument("--test", help="test the model")
    action.add_argument("--text", help="predict the output for the given text")

    parser.add_argument("--path", help="path to the model to load")
    parser.add_argument("--rawData", action="store_true", help="print raw output from the model", default=False)

    parser.add_argument("--save", help="save the model to the given path")
    parser.add_argument("--plot", action="store_true", help="plot the training data")
    parser.add_argument("--epochs", type=int, help="number of epochs to train for", default=EPOCHS)
    parser.add_argument("--batchSize", type=int, help="batch size", default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, help="learning rate", default=LR)

    parser.add_argument("--hideTfLogs", action="store_true", help="hide TensorFlow logs", default=True)
    parser.add_argument("--threshold", type=float, help="threshold for the NER logits", default=0.5)

    args = parser.parse_args()

    if args.hideTfLogs:
        import os
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        warnings.filterwarnings("ignore")

    from model import MTLModel

    model = MTLModel(MAP_PATH)
    
    if args.train:
        from train import train_model

        model, classification_loss, entity_loss, total_loss = train_model(
            model,
            save_to=args.save,
            epochs=args.epochs,
            batch_size=args.batchSize,
            lr=args.lr,
        )

        if args.plot:
            from utils import plot_training_data

            plot_training_data(
                classification_loss,
                entity_loss,
                total_loss,
            )
    elif args.test:
        raise NotImplementedError
    elif args.text:
        from utils import predict
        import torch

        print(f"Loading {'model from ' + args.path if args.path else 'default model'}...")

        if args.path:
            model.load_state_dict(torch.load(args.path))
        
        print(f"Processing text: '{args.text}' with threshold {args.threshold}")

        classification_probs, ner_logits, entity_labels = predict(
            args.text,
            model,
            threshold=args.threshold,
        )

        if args.rawData:
            print("[RAW] Classification probabilities:", classification_probs)
            print("[RAW] NER logits:", ner_logits)
            print("[RAW] Entity labels:", entity_labels)
        
        """
        == Classes ==
        Temperature             0.923 |||||||||
        Lights                  0.100 |
        Blinds                  0.002

        == Entities ==
        Number          70      0 - 0
        """
        print("== Classes ==")
        for classification, prob in zip(model.mapping.classification_map["id_to_label"].values(), classification_probs.tolist()[0]):
            print(f"{classification: <20}{prob:.3f} {'|' * int(prob * 10)}")
        print()
        print("== Entities ==")
        # entity_labels example: [('number', (1, 1)), ('numbernumber', (3, 4))]
        for entity, (start, end) in entity_labels:
            name = f"{entity[:14]: <15}"
            span = f"{start} - {end}"
            # clip value
            value = args.text[start:end + 1][:7]
            print(f"{name}{value: <8}{span}")

        if len(entity_labels) == 0:
            print("No entities identified")
    else:
        print("No action specified")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())