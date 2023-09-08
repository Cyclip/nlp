import sys
import argparse
from model import MTLModel
from constants import MAP_PATH, EPOCHS, BATCH_SIZE, LR

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

    args = parser.parse_args()

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

        if args.path:
            model.load_state_dict(torch.load(args.path))

        classification_output, ner_output, ner_span, loss = predict(
            model,
            args.text,
        )

        if args.rawData:
            print("[Raw] Classification output:", classification_output)
            print("[Raw] NER output:", ner_output)
            print("[Raw] NER span:", ner_span)

        # Convert the output to classification label idx, NER label idx
        classification_label = model.mapping.get_class_label(classification_output)
        ner_label = model.mapping.get_entity_label(ner_output)

        print("Classification label:", classification_label)
        print("NER label:", ner_label)
        print("NER span:", ner_span)
        print("Loss:", loss)
    else:
        print("No action specified")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())