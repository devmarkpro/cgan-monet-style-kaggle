import argparse
import logging

from app import App
from logger import log, Colors
import time
import dotenv
from configs import AppParams

dotenv.load_dotenv()


def main(params):
    log(msg="starting the app", color=Colors.BLUE)
    App(params)()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for reproducibility, default: 42"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="number of epochs for training, default: 100"
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
        help="Logging level, default: INFO",
    )
    parser.add_argument(
        "--log_file", type=str, default="./app.log", help="Full path to log file, default: ./app.log"
    )
    parser.add_argument(
        "--ascii_art",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether to output ascii art, default: 1",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./data/monet/training",
        help="Path to the dataset directory, default: ./data/monet/training",
    )
    parser.add_argument("--workers", type=int, default=1, help="Number of workers, default: 1")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size, default: 128")
    parser.add_argument(
        "--artifacts_folder",
        type=str,
        default="./artifacts",
        help="Path to store the plots, default: ./artifacts",
    )
    parser.add_argument(
        "--use_wandb",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether to use wandb or not, if set to 1, you have to enter your wandb API key in the .env, default: 1",
    )
    parser.add_argument(
        "--num_channels",
        type=int,
        default=3,
        help="Number of channels in the training images. For color images this is 3, default: 3",
    )
    parser.add_argument(
        "--latent_size",
        type=int,
        default=100,
        help="Size of z latent vector (i.e. size of generator input), default: 100",
    )
    parser.add_argument(
        "--generator_feature_map_size",
        type=int,
        default=64,
        help="Size of feature maps in generator, default: 64",
    )
    parser.add_argument(
        "--discriminator_feature_map_size",
        type=int,
        default=64,
        help="Size of feature maps in discriminator, default: 64",
    )
    parser.add_argument(
        "--discriminator_lr",
        type=float,
        default=0.0002,
        help="Learning rate for discriminator optimizer, default: 0.0002",
    )
    parser.add_argument(
        "--generator_lr",
        type=float,
        default=0.0002,
        help="Learning rate for generator optimizer, default: 0.0002",
    )
    parser.add_argument(
        "--discriminator_beta1",
        type=float,
        default=0.5,
        help="Beta1 hyperparameter for Adam optimizers, default: 0.5",
    )
    parser.add_argument(
        "--generator_beta1",
        type=float,
        default=0.5,
        help="Beta1 hyperparameter for Adam optimizers, default: 0.5",
    )
    parser.add_argument(
        "--label_smoothing_real",
        type=float,
        default=1.0,
        help="Target label for real samples in discriminator loss; 1.0 disables smoothing (e.g., 0.9)",
    )
    parser.add_argument(
        "--mifid_eval_every_epochs",
        type=int,
        default=1,
        help="Number of epochs between MIFID evaluations, default: 1",
    )
    parser.add_argument(
        "--mifid_eval_batches",
        type=int,
        default=5,
        help="Number of batches to evaluate MIFID on, default: 5",
    )
    parser.add_argument(
        "--image_log_every_iters",
        type=int,
        default=10,
        help="Number of iterations between image logging, default: 10",
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
    # Normalize CLI args to AppParams dataclass for consistent WandB config logging
    params = AppParams(
        seed=args.seed,
        log_level=args.log_level,
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        workers=args.workers,
        epochs=args.epochs,
        artifacts_folder=args.artifacts_folder,
        use_wandb=bool(args.use_wandb),
        num_channels=args.num_channels,
        latent_size=args.latent_size,
        generator_feature_map_size=args.generator_feature_map_size,
        discriminator_feature_map_size=args.discriminator_feature_map_size,
        discriminator_lr=args.discriminator_lr,
        generator_lr=args.generator_lr,
        discriminator_beta1=args.discriminator_beta1,
        generator_beta1=args.generator_beta1,
        label_smoothing_real=args.label_smoothing_real,
        mifid_eval_every_epochs=args.mifid_eval_every_epochs,
        mifid_eval_batches=args.mifid_eval_batches,
        image_log_every_iters=args.image_log_every_iters,
    )
    main(params)
    end_time = time.time()
    elapsed_time = end_time - start_time

    minutes, seconds = divmod(elapsed_time, 60)
    log(msg=f"execution time: {int(minutes)}m {seconds:.2f}s", color=Colors.GREEN)
