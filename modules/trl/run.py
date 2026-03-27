import sys
from perl.train import train
from perl.utils.utils import parse_args_to_config

if __name__ == "__main__":
    config = parse_args_to_config()
    train(config)