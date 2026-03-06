import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset", default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("-nhl", "--num_layers", type=int, default=3)
    parser.add_argument("-sz", "--hidden_size", type=int, nargs="+", default=[128])
    parser.add_argument("-a", "--activation", default="relu")
    parser.add_argument("-l", "--loss", default="cross_entropy")
    parser.add_argument("-o", "--optimizer", default="rmsprop")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-w_i", "--weight_init", default="xavier")
    parser.add_argument("-e", "--epochs", type=int, default=20)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-w_p", "--wandb_project", default=None)
    parser.add_argument("--model_path", default="best_model.npy")
    parser.add_argument("--config_path", default="best_config.json")

    args, _ = parser.parse_known_args()

    if isinstance(args.hidden_size, list) and len(args.hidden_size) == 1:
        args.hidden_size = args.hidden_size[0]

    return args


def load_weights(path):
    return np.load(path, allow_pickle=True).item()


def main():

    args = parse_arguments()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(BASE_DIR, args.config_path)

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            setattr(args, k, v)
        print(f"Loaded config from {config_path}")

    _, _, X_test, y_test = load_data(args.dataset)

    model = NeuralNetwork(args)

    model_path = os.path.join(BASE_DIR, args.model_path)
    weights = load_weights(model_path)
    model.set_weights(weights)
    print(f"Loaded weights from {model_path}")

    results = model.evaluate(X_test, y_test)

    print(f"Accuracy  : {results['accuracy']:.4f}")
    print(f"F1-Score  : {results['f1']:.4f}")
    print(f"Precision : {results['precision']:.4f}")
    print(f"Recall    : {results['recall']:.4f}")
    print(f"Loss      : {results['loss']:.4f}")


if __name__ == "__main__":
    main()