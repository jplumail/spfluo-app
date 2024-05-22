import os


def main():
    os.environ["SCIPION_HOME"] = os.path.expanduser(os.path.join("~","scipion"))
    os.environ["SCIPION_USER_DATA"] = os.path.expanduser(os.path.join("~","ScipionFluoUserData"))

    if not os.path.exists(os.environ["SCIPION_HOME"]):
        os.mkdir(os.environ["SCIPION_HOME"])

    from scipion.__main__ import main as scipion_main

    scipion_main()