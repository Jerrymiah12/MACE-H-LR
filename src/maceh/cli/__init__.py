"""Unified command-line interface for the reusable MACE-H operations."""

import argparse
import os


def _set_threads(count):
    if count is None:
        return
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = str(count)
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(count)


def _run_kernel(action, args):
    _set_threads(args.threads)
    from maceh import DeepHE3Kernel
    kernel = DeepHE3Kernel()
    if action == "eval":
        kernel.eval(args.config, debug=args.debug)
    else:
        getattr(kernel, action)(args.config)
    return 0


def _run_epc(args):
    _set_threads(args.threads)
    from maceh.epc.run import run_epc
    run_epc(args.config, debug=args.debug)
    return 0


def _add_common(parser):
    parser.add_argument("config", metavar="CONFIG", help="INI configuration file")
    parser.add_argument("-n", "--threads", type=int,
                        help="maximum CPU thread count")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="maceh", description="MACE-H-LR training and response tools")
    commands = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
            ("preprocess", "preprocess DFT structures into graph data"),
            ("train", "train a Hamiltonian model")):
        command = commands.add_parser(name, help=help_text)
        _add_common(command)
        command.set_defaults(handler=lambda args, action=name:
                             _run_kernel(action, args))

    evaluate = commands.add_parser("eval", help="evaluate a trained model")
    _add_common(evaluate)
    evaluate.add_argument("--debug", action="store_true",
                          help="fill unpredicted matrix elements with zero")
    evaluate.set_defaults(handler=lambda args: _run_kernel("eval", args))

    epc_help = "compute Cartesian-AO electron-phonon coupling"
    epc = commands.add_parser("epc", help=epc_help, description=epc_help)
    _add_common(epc)
    epc.add_argument("--debug", action="store_true",
                     help="fill unpredicted matrix elements with zero")
    epc.set_defaults(handler=_run_epc)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.handler(args)


__all__ = ["build_parser", "main"]
