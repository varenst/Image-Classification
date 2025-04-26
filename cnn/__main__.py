import argparse
from cnn.modeltraining import train_pytorch_cnn
from cnn.testing import test

argparser = argparse.ArgumentParser(description="Run a CNN model for image classification.")
argparser.add_argument("--train", action="store_true", help="trains a CNN")
argparser.add_argument("--test", action="store_true", help="tests model")
parsed_arguments  = argparser.parse_args()

if parsed_arguments.train:
   train_pytorch_cnn.train()
elif parsed_arguments.test:
   test.test()
else:
    print("Please specify --train or --test")
