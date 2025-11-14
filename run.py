import sys
from open_tinker.train import train
from open_tinker.utils.utils import parse_args_to_config

if __name__ == "__main__":
    config = parse_args_to_config()
    train(config)