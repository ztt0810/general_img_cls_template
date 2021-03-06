from tools import Trainer
from tools import Inference
from config import Config
import argparse


def cmd_param():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=bool, default=True, help='start training')
    parser.add_argument('--test', type=bool, default=True, help='test and get the result')
    parser.add_argument('--use_tta', type=bool, default=False, help='use test time augmentation')
    # parser.add_argument('--mix_prec', type=bool, default=False, help='train with mixed precision')
    return parser.parse_args()


def _main():
    opt = cmd_param()
    if opt.train:
        trainer = Trainer()
        trainer.run_train()
    if opt.test:
        if opt.use_tta:
            pass
        else:
            infer = Inference(Config.load_model)
            infer.predict()


if __name__ == '__main__':
    _main()