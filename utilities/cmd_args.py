import argparse, yaml, sys


def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='Get arguments from bash file')

    parser.add_argument('--cfg',
                        dest='cfg_file',
                        type=str,
                        required=True,
                        help="configuration file *.yml")
    parser.add_argument('--repeat',
                        type=int,
                        default=1,
                        help='The number of repeated jobs.')
    parser.add_argument('--mark_done',
                        action='store_true',
                        help='Mark yaml as done after a job has finished.')
    parser.add_argument('opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='See run/config_HinSAGE.py for remaining options.')  
    return parser.parse_args()
