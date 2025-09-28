import argparse
import logging

from app import App
from logger import log, Colors
import time
import dotenv

dotenv.load_dotenv()


def main(params):
    log(msg="starting the app", color=Colors.BLUE)
    App(params)()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for reproducibility"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="number of epochs for training"
    )
    parser.add_argument(
        "--log_level",
        type=int,
        default=logging.INFO,
        choices=[
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ],
        help="Logging level",
    )
    parser.add_argument(
        "--log_file", type=str, default="./app.log", help="Full path to log file"
    )
    parser.add_argument(
        "--ascii_art",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether to output ascii art",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./data/monet/training",
        help="Path to the dataset directory",
    )
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--artifacts_folder",
        type=str,
        default="./artifacts",
        help="Path to store the plots",
    )
    parser.add_argument(
        "--use_wandb",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether to use wandb or not, if set to 1, you have to enter your wandb API key in the .env",
    )
    parser.add_argument(
        "--num_channels",
        type=int,
        default=3,
        help="Number of channels in the training images. For color images this is 3",
    )
    parser.add_argument(
        "--latent_size",
        type=int,
        default=100,
        help="Size of z latent vector (i.e. size of generator input)",
    )
    parser.add_argument(
        "--generator_feature_map_size",
        type=int,
        default=64,
        help="Size of feature maps in generator",
    )
    parser.add_argument(
        "--discriminator_feature_map_size",
        type=int,
        default=64,
        help="Size of feature maps in discriminator",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0002,
        help="Learning rate for optimizers",
    )
    parser.add_argument(
        "--discriminator_beta1",
        type=float,
        default=0.999,
        help="Beta1 hyperparameter for Adam optimizers",
    )
    parser.add_argument(
        "--generator_beta1",
        type=float,
        default=0.999,
        help="Beta1 hyperparameter for Adam optimizers",
    )
    return parser.parse_args()


if __name__ == "__main__":
    start_time = time.time()

    args = parse_arguments()
    show_ascii_art: bool = True if args.ascii_art == 1 else False

    if show_ascii_art:
        print(
            f"""
▗▄▄▄   ▗▄▄▖ ▗▄▄▖ ▗▄▖ ▗▖  ▗▖        ▗▖  ▗▖ ▄▄▄  ▄▄▄▄  ▗▞▀▚▖   ■  
▐▌  █ ▐▌   ▐▌   ▐▌ ▐▌▐▛▚▖▐▌        ▐▛▚▞▜▌█   █ █   █ ▐▛▀▀▘▗▄▟▙▄▖
▐▌  █ ▐▌   ▐▌▝▜▌▐▛▀▜▌▐▌ ▝▜▌        ▐▌  ▐▌▀▄▄▄▀ █   █ ▝▚▄▄▖  ▐▌  
▐▙▄▄▀ ▝▚▄▄▖▝▚▄▞▘▐▌ ▐▌▐▌  ▐▌        ▐▌  ▐▌                   ▐▌  
                                                            ▐▌  
    """
        )
    main(args)
    end_time = time.time()
    elapsed_time = end_time - start_time

    minutes, seconds = divmod(elapsed_time, 60)
    log(msg=f"execution time: {int(minutes)}m {seconds:.2f}s", color=Colors.GREEN)
