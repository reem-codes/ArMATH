import argparse
import os

from train_iteration import *

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size",
                    default=64,
                    type=int,
                    help="Number of training samples per batch.")
parser.add_argument("--embedding-size",
                    default=128,
                    type=int,
                    help="Size of embedding layers.")
parser.add_argument("--hidden-size",
                    default=512,
                    type=int,
                    help="Size of hidden layers.")
parser.add_argument("--n-epochs",
                    default=80,
                    type=int,
                    help="Number of epochs of training")
parser.add_argument("--learning-rate",
                    default=1e-3,
                    type=float,
                    help="Learning rate.")
parser.add_argument("--weight-decay",
                    default=1e-5,
                    type=float,
                    help="Weight decay.")
parser.add_argument("--beam-size",
                    default=5,
                    type=int,
                    help="Beam size")
parser.add_argument("--trim-min-count",
                    default=5,
                    type=int,
                    help="Min number of word count to keep")
parser.add_argument("--n-layers",
                    default=2,
                    type=int,
                    help="Number of layers")
parser.add_argument("--n-workers",
                    type=int,
                    default=4,
                    help="number of CPUs for data loader workers.")
parser.add_argument("--seed",
                    type=int,
                    default=354,
                    help="Seed used for pseudorandom number generation.")
parser.add_argument("--output-dir",
                    type=str,
                    default="results/",
                    help="Path to directory where output should be written.")
parser.add_argument("--embedding-type",
                    type=str,
                    default="one-hot",
                    help="[one-hot, aravec, fasttext]")
parser.add_argument("--embedding-model-name",
                    type=str,
                    help="path to word2vec model")
parser.add_argument("--train-one-fold-only",
                    action='store_true',
                    help="True - train with one fold, test with 4 (for ablation). False (default) - train with 4 folds, test with one")
parser.add_argument("--arabic",
                    action='store_true',
                    help="Is this for training arabic or Chinese?")
parser.add_argument("--data-path",
                    type=str,
                    default="datasets/armath",
                    help="filepath of datafile (chinese) or datafolder (arabic)")
parser.add_argument("--transfer-learning",
                    action='store_true',
                    help="True - Train with transfer learning. False - train GTS (no transfer learning)")
parser.add_argument("--transfer-learning-model",
                    type=str,
                    help="path to transfer learning model")
parser.add_argument("--transfer-learning-transfer-encoder",
                    action='store_true',
                    help="Should encoder weights be copied?")
parser.add_argument("--transfer-learning-transfer-decoder",
                    action='store_true',
                    help="Should decoder weights be copied?")
parser.add_argument("--evaluate",
                    action='store_true',
                    help="evaluate the model")
parser.add_argument("--config-path",
                    type=str,
                    help="path to config file of the trained model")

config = vars(parser.parse_args())

if not config["output_dir"].endswith("/"):
    config["output_dir"] += "/"

try:
    os.makedirs(config["output_dir"])
except OSError:
    pass

if config["evaluate"]:
    evaluate_model(config)
else:
    train_model(config)
