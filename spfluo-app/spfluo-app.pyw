import os


scipion_path = os.path.expanduser(os.path.join("~","scipion"))
os.environ["SCIPION_HOME"] = scipion_path
os.environ["SCIPION_USER_FLUO_DATA"] = os.path.expanduser(os.path.join("~","ScipionUserFluoData"))
os.environ["SCIPION_USER_DATA"] = os.environ["SCIPION_USER_FLUO_DATA"]

if not os.path.exists(os.environ["SCIPION_HOME"]):
    os.mkdir(os.environ["SCIPION_HOME"])

from scipion.__main__ import main

main()