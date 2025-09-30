import argparse
import logging
import sys

from app import App
from eda import EDA
from logger import log, Colors
import time
import dotenv
from configs import AppParams

dotenv.load_dotenv()


def main(params):
    log(msg="starting the app", color=Colors.BLUE)
    App(params)()


def create_base_parser():
    """Create base parser with common arguments"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for reproducibility, default: 42"
    )
    parser.add_argument(
        "--log_level",
        type=int,
        default=logging.INFO,
        help="log level, default: 20 (INFO)",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./data/monet/training/monet_jpg",
        help="Path to the dataset directory, default: ./data/monet/training/monet_jpg",
    )
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of workers, default: 0")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size, default: 16")
    parser.add_argument(
        "--artifacts_folder",
        type=str,
        default="./artifacts",
        help="Path to store the plots, default: ./artifacts",
    )
    return parser


def add_gan_arguments(parser):
    parser.add_argument(
        "--epochs", type=int, default=100, help="number of epochs for training, default: 100"
    )
    parser.add_argument(
        "--ascii_art",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether to output ascii art, default: 1",
    )
    parser.add_argument(
        "--use_wandb",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether to use wandb for logging, default: 1",
    )
    parser.add_argument(
        "--num_channels", type=int, default=3, help="Number of channels in the training images, default: 3"
    )
    parser.add_argument(
        "--latent_size", type=int, default=128, help="Size of latent vector, default: 128"
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
        help="One-sided label smoothing for real labels, default: 1.0 (disabled)",
    )
    parser.add_argument(
        "--mifid_eval_every_epochs",
        type=int,
        default=0,
        help="Evaluate MiFID every N epochs, default: 0 (disabled for performance)",
    )
    parser.add_argument(
        "--mifid_eval_batches",
        type=int,
        default=4,
        help="Number of batches to use for MiFID evaluation, default: 4 (all batches for small dataset)",
    )
    parser.add_argument(
        "--image_log_every_iters",
        type=int,
        default=10,
        help="Log images every N iterations, default: 10",
    )


def add_eda_arguments(parser):
    """Add EDA-specific arguments to parser"""
    parser.add_argument(
        "--output_format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format for plots, default: png",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="Number of images to sample for analysis, default: 100 (0 = all images)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved plots, default: 150",
    )


def parse_arguments():
    """Parse command line arguments with subcommands"""
    # Main parser
    main_parser = argparse.ArgumentParser(
        description="DCGAN Monet Style Transfer")
    subparsers = main_parser.add_subparsers(
        dest='command', help='Available commands')

    # Create base parser for common arguments
    base_parser = create_base_parser()

    # Add GAN subcommand
    gan_parser = subparsers.add_parser(
        'gan', parents=[base_parser], help='Train DCGAN model')
    add_gan_arguments(gan_parser)

    # Add EDA subcommand
    eda_parser = subparsers.add_parser(
        'eda', parents=[base_parser], help='Perform exploratory data analysis')
    add_eda_arguments(eda_parser)

    args = main_parser.parse_args()

    # If no command specified, show help
    if args.command is None:
        main_parser.print_help()
        sys.exit(1)

    return args


if __name__ == "__main__":
    start_time = time.time()

    args = parse_arguments()

    if args.command == 'gan':
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

    elif args.command == 'eda':
        log(msg="Starting EDA analysis", color=Colors.BLUE)
        # Create EDA params (only need common parameters)
        params = AppParams(
            seed=args.seed,
            log_level=args.log_level,
            dataset_dir=args.dataset_dir,
            batch_size=args.batch_size,
            workers=args.workers,
            artifacts_folder=args.artifacts_folder,
        )
        eda = EDA(params)
        eda.run_full_analysis()
        eda.generate_summary_report()
        log(msg="EDA analysis completed!", color=Colors.GREEN)
    end_time = time.time()
    elapsed_time = end_time - start_time

    minutes, seconds = divmod(elapsed_time, 60)
    log(msg=f"execution time: {int(minutes)}m {seconds:.2f}s",
        color=Colors.GREEN)
