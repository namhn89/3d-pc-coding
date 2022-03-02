import argparse
import os

from pcc_model import PCCModel
from trainer import Trainer
from utils.general import init_log


def parse_args():
    parser = argparse.ArgumentParser(description="Compression Model Training")
    parser.add_argument("--path_config", type=str, help="Path to the configuration")
    return parser.parse_args()


def main():
    opt = parse_args()
    model = PCCModel()
    init_log(str(os.path.join(os.path.dirname(opt.path_config), "train.log")))
    trainer = Trainer(opt.path_config, model)
    trainer.run()


if __name__ == "__main__":
    main()
