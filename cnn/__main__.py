import argparse
import sys

argparser = argparse.ArgumentParser(description="Run a CNN model for image classification.")
argparser.add_argument("--train", action="store_true", help="trains a CNN")
argparser.add_argument("--test", action="store_true", help="tests model")
parsed_arguments  = argparser.parse_args()

if parsed_arguments.train:
    print("training")
elif parsed_arguments.test:
    print("testing")
else:
    print("Please specify --train or --test")
