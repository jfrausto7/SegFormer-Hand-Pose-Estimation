
import argparse


def parse_args() -> argparse.Namespace:
    """Parse arguments from command line into ARGS."""

    parser = argparse.ArgumentParser(
        description="The runner for our BezierModel model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--train',
        help='Train',
        action='store_true',
        dest='train'
    )

    parser.add_argument(
        '--test',
        help='Test',
        action='store_true',
        dest='test'
    )

    parser.add_argument(
        '--weights',
        default='',
        help='Path for the weights to use in training/testing',
        dest='weights'
    )

    parser.add_argument(
        '--epochs',
        default=10,
        type=int,
        help='Number of training epochs',
        dest='epochs'
    )

    parser.add_argument(
        '--visualize',
        help='Save a visualization of model architecture',
        action='store_true',
        dest='viz'
    )

    return parser.parse_args()
