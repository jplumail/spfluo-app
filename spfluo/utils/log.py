import argparse
import logging

logging.basicConfig()
spfluo_logger = logging.getLogger("spfluo")

base_parser = argparse.ArgumentParser(
    "spfluo cli tool",
    add_help=False,
)
base_parser.add_argument("--debug", action="store_true")


def set_logging_level(args: argparse.Namespace):
    debug_arg, _ = base_parser.parse_known_args(namespace=args)
    if debug_arg.debug:
        spfluo_logger.setLevel(logging.DEBUG)
