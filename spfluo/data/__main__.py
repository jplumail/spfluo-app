import os
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

from spfluo.conftest import DATA_DIR


def main(args):
    if args.get_path:
        # Run generation if not done
        test_file = (
            Path(__file__).parent.parent
            / "picking/modules/pretraining/generate/tests/test_data_generator.py"
        )
        subprocess.call(
            ["python", "-m", "pytest", str(test_file.absolute()) + "::test_generation"],
            stdout=open(os.devnull, "w"),
        )
        # Write path to stdout
        sys.stdout.write(str(DATA_DIR.absolute()))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--get_fixtures_path", action="store_true")
    main(parser.parse_args())
