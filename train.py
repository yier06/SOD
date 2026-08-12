"""SODA-D 目标检测训练入口。"""

import argparse

from config.dataset.config import Config
from detection.engine.train import train


def main():
    parser = argparse.ArgumentParser(description="Train CustomYOLO11 on SODA-D")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--scale", choices=["n", "s", "m", "l", "x"], default=None)
    parser.add_argument("--save-dir", default=None)
    args = parser.parse_args()

    config = Config()
    overrides = {
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "num_workers": args.workers,
        "model_scale": args.scale,
        "save_dir": args.save_dir,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(config, name, value)
    train(config)


if __name__ == "__main__":
    main()
