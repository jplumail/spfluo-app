import os

os.environ["SCIPION_HOME"] = "/home/plumail/new_scipion"

if not os.path.exists(os.environ["SCIPION_HOME"]):
    os.mkdir(os.environ["SCIPION_HOME"])

from scipion.__main__ import main

main()