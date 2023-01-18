import os
import argparse


def parse_args() -> argparse.Namespace:
    help_msg = 'Folder above the figtree source. Defaults to the setup.py directory.'
    parser = argparse.ArgumentParser('Setup pyfigtree library usage for data generation')
    parser.add_argument('-r', '--rootdir', type=str, default=None, help=help_msg)
    parser.add_argument('-a', '--action', type=str, default='setup', choices=['setup', 'check'])
    return parser.parse_args()


def setup(rootdir: str=None) -> None:
    if rootdir is None:
        rootdir = os.path.dirname(os.path.abspath(__file__))  # abs path to setup.py folder.
    figtree_path = os.path.join(rootdir, "figtree", "figtree-master")
    pyfigtree_path = os.path.join(rootdir, "figtree", "pyfigtree")
    exports = ("\n" +
               "# pyfigtree \n"
               f"export FIGTREEDIR={figtree_path} \n"
               "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$FIGTREEDIR/lib \n"
               f"export PYTHONPATH={pyfigtree_path}:$PYTHONPATH \n")
    os.system(f"echo '{exports}' >> ~/.bashrc")
    os.chdir(figtree_path)
    os.system("make")
    os.system("clear")
    os.system("echo 'Figtree setup completed.'")


def check(rootdir: str=None) -> None:
    if rootdir is None:
        rootdir = os.path.dirname(os.path.abspath(__file__))  # abs path to setup.py folder
    pyfigtree_path = os.path.join(rootdir, "figtree", "pyfigtree")
    os.chdir(pyfigtree_path)
    os.system("python pyfigtree.py")


if __name__ == '__main__':
    args = parse_args()
    function = setup if args.action == 'setup' else check
    function(args.rootdir)
