import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dptb.entrypoints.train import train
from dptb.entrypoints.config import config
from dptb.entrypoints.test import _test
from dptb.entrypoints.run import run
from dptb.entrypoints.bond import bond
from dptb.entrypoints.nrl2json import nrl2json
from dptb.entrypoints.pth2json import pth2json
from dptb.entrypoints.data import data
from dptb.utils.loggers import set_log_handles
from dptb.utils.config_check import check_config_train
from dptb.entrypoints.collectskf import skf2pth, skf2nnsk
from dptb.entrypoints.emp_sk import to_empsk

from dptb import __version__



def get_ll(log_level: str) -> int:
    """Convert string to python logging level.

    Parameters
    ----------
    log_level : str
        allowed input values are: DEBUG, INFO, WARNING, ERROR, 3, 2, 1, 0

    Returns
    -------
    int
        one of python logging module log levels - 10, 20, 30 or 40
    """
    if log_level.isdigit():
        int_level = (4 - int(log_level)) * 10
    else:
        int_level = getattr(logging, log_level)

    return int_level

def main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DeepTB: A deep learning package for Tight-Binding Model"
                    " with first-principle accuracy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('-v', '--version', 
                        action='version', version=f'%(prog)s {__version__}', help="show the DeepTB's version number and exit")


    subparsers = parser.add_subparsers(title="Valid subcommands", dest="command")

    # log parser
    parser_log = argparse.ArgumentParser(
        add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_log.add_argument(
        "-ll",
        "--log-level",
        choices=["DEBUG", "3", "INFO", "2", "WARNING", "1", "ERROR", "0"],
        default="INFO",
        help="set verbosity level by string or number, 0=ERROR, 1=WARNING, 2=INFO "
             "and 3=DEBUG",
    )

    parser_log.add_argument(
        "-lp",
        "--log-path",
        type=str,
        default=None,
        help="set log file to log messages to disk, if not specified, the logs will "
             "only be output to console",
    )

    # config parser
    parser_config = subparsers.add_parser(
        "config",
        parents=[parser_log],
        help="get config templete",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser_config.add_argument(
        "PATH", help="the path you want to put the config templete in",
        type=str,
        default="./input_templete.json"
    )

    parser_config.add_argument(
        "-m", 
        "--model",
        type=str,
        default=None,
        help="load model to update input template."
    )

    parser_config.add_argument(
        "-tr", 
        "--train",
        help="Generate the config templete for training.",
        action="store_true"
    )
    
    parser_config.add_argument(
        "-ts", 
        "--test",
        help="Generate the config templete for testing.",
        action="store_true"
    )

    parser_config.add_argument(
        "-e3", 
        "--e3tb",
        help="Generate the config templete for e3nn TB model.",
        action="store_true"
    )

    parser_config.add_argument(
        "-sk", 
        "--sktb",
        help="Generate the config templete for nn-sk TB model.",
        action="store_true"
    )

    parser_config.add_argument(
        "-skenv", 
        "--sktbenv",
        help="Generate the config templete for nn-sk env TB model.",
        action="store_true"
    )

    # neighbour
    parser_bond = subparsers.add_parser(
        "bond",
        parents=[parser_log],
        help="Bond distance analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser_bond.add_argument(
        "struct", help="the structure input, must be ase readable structure format",
        type=str,
        default="./POSCAR"
    )

    parser_bond.add_argument(
        "-acc",
        "--accuracy",
        type=float,
        default=1e-3,
        help="The accuracy to judge whether two bond are the same.",
    )

    parser_bond.add_argument(
        "-c",
        "--cutoff",
        type=float,
        default=6.0,
        help="The cutoff radius of bond search.",
    )

    # nrl2json
    parser_nrl2json = subparsers.add_parser(
        "n2j",
        parents=[parser_log],
        help="Convert the NRL file to json ckpt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_nrl2json.add_argument(
        "INPUT", help="the input parameter file in json or yaml format",
        type=str,
        default=None   
    )
    parser_nrl2json.add_argument(
        "-nrl",
        "--nrl_file",
        type=str,
        default=None,
        help="The NRL file name"
    )
    parser_nrl2json.add_argument(
        "-o",
        "--outdir",
        type=str,
        default="./",
        help="The output files to save the transfered model and updated input."
    )

    # pth2json
    parser_pth2json = subparsers.add_parser(
        "p2j",
        parents=[parser_log],
        help="Convert the PTH ckpt to json ckpt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_pth2json.add_argument(
        "-i",
        "--init-model",
        type=str,
        default=None,
        help="The pth ckpt to be transfered to json.",
    )
    parser_pth2json.add_argument(
        "-o",
        "--outdir",
        type=str,
        default="./",
        help="The output files to save the transfered model."
    )

    # train parser
    parser_train = subparsers.add_parser(
        "train",
        parents=[parser_log],
        help="train a model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser_train.add_argument(
        "INPUT", help="the input parameter file in json or yaml format",
        type=str,
        default=None
    )
    parser_train.add_argument(
        "-i",
        "--init-model",
        type=str,
        default=None,
        help="Initialize the model by the provided checkpoint.",
    )
    
    parser_train.add_argument(
        "-r",
        "--restart",
        type=str,
        default=None,
        help="Restart the training from the provided checkpoint.",
    )


    parser_train.add_argument(
        "-o",
        "--output",
        type=str,
        default="./",
        help="The output files in training.",
    )

    parser_test = subparsers.add_parser(
        "test",
        parents=[parser_log],
        help="test the model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser_test.add_argument(
        "INPUT", help="the input parameter file in json or yaml format",
        type=str,
        default=None
    )
    
    parser_test.add_argument(
        "-i",
        "--init-model",
        type=str,
        default=None,
        help="Initialize the model by the provided checkpoint.",
    )

    parser_test.add_argument(
        "-o",
        "--output",
        type=str,
        default="./",
        help="The output files in testing.",
    )

    
    parser_run = subparsers.add_parser(
        "run",
        parents=[parser_log],
        help="run the TB with a model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser_run.add_argument(
        "INPUT", help="the input parameter file for postprocess run in json format",
        type=str,
        default=None
    )

    parser_run.add_argument(
        "-i",
        "--init-model",
        type=str,
        default=None,
        help="Initialize the model by the provided checkpoint.",
    )

    parser_run.add_argument(
        "-stu",
        "--structure",
        type=str,
        default=None,
        help="the structure file name wiht its suffix of format, such as, .vasp, .cif etc., prior to the model_ckpt tags in the input json. "
    )

    parser_run.add_argument(
        "-o",
        "--output",
        type=str,
        default="./",
        help="The output files in postprocess run."
    )

    # preprocess data
    parser_data = subparsers.add_parser(
        "data",
        parents=[parser_log],
        help="preprocess software output",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser_data.add_argument(
        "INPUT", help="the input parameter file in json or yaml format",
        type=str,
        default=None
    )

    parser_data.add_argument(
        "-p",
        "--parse",
        action="store_true",
        help="Initialize the training from the frozen model.",
    )

    parser_data.add_argument(
        "-s",
        "--split",
        action="store_true",
        help="Initialize the training from the frozen model.",
    )

    parser_data.add_argument(
        "-c",
        "--collect",
        action="store_true",
        help="Initialize the training from the frozen model.",
    )

        # preprocess data
    parser_cskf = subparsers.add_parser(
        "cskf",
        parents=[parser_log],
        help="collect the sktb params from sk files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser_cskf.add_argument(
        "-d",
        "--dir_path",
        type=str,
        default="./",
        help="The directory of the sk files."
    )

    parser_cskf.add_argument(
        "-o",
        "--output",
        type=str,
        default="skparams.pth",
        help="The output pth files of sk params from skfiles."
    )

    # neighbour
    parser_skf2nn = subparsers.add_parser(
        "skf2nn",
        parents=[parser_log],
        help="Convert the sk files to nn-sk TB model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser_skf2nn.add_argument(
        "INPUT", help="the input parameter file in json or yaml format",
        type=str,
        default=None
    )

    parser_skf2nn.add_argument(
        "-i",
        "--init-model",
        type=str,
        default=None,
        help="Initialize the model by the provided checkpoint.",
    )

    parser_skf2nn.add_argument(
        "-o",
        "--output",
        type=str,
        default="./",
        help="The output files in training.",
    )

    parser_esk = subparsers.add_parser(
        "esk",
        parents=[parser_log],
        help="Generate initial empirical SK parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_esk.add_argument(
        "INPUT", help="the input parameter file in json or yaml format",
        type=str,
        default=None
    )
    parser_esk.add_argument(
        "-o",
        "--output",
        type=str,
        default="./",
        help="The output files in training."
    )
    parser_esk.add_argument(
        "-m",
        "--basemodel",
        type=str,
        default="poly2",
        help="The base model type can be poly2 or poly4."
    )
    return parser

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse arguments and convert argument strings to objects.

    Parameters
    ----------
    args: List[str]
        list of command line arguments, main purpose is testing default option None
        takes arguments from sys.argv

    Returns
    -------
    argparse.Namespace
        the populated namespace
    """
    parser = main_parser()
    parsed_args = parser.parse_args(args=args)
    if parsed_args.command is None:
        parser.print_help()
    else:
        parsed_args.log_level = get_ll(parsed_args.log_level)

    return parsed_args

def main():
    args = parse_args()

    if args.command not in (None, "train", "test", "run"):
        set_log_handles(args.log_level, Path(args.log_path) if args.log_path else None)

    dict_args = vars(args)
    
    if args.command == 'config':
        config(**dict_args)

    elif args.command == 'bond':
        bond(**dict_args)

    elif args.command == 'train':
        check_config_train(**dict_args)
        train(**dict_args)

    elif args.command == 'test':
        _test(**dict_args)

    elif args.command == 'run':
        run(**dict_args)

    elif args.command == 'n2j':
        nrl2json(**dict_args)
    
    elif args.command == 'p2j':
        pth2json(**dict_args)

    elif args.command == 'data':
        data(**dict_args)
        
    elif args.command == 'cskf':
        skf2pth(**dict_args)

    elif args.command == 'skf2nn':
        skf2nnsk(**dict_args)

    elif args.command == 'esk':
        to_empsk(**dict_args)
