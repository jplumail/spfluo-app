import pathlib

import pyworkflow as pw
from flask import Flask, abort, send_file

app = Flask(__name__, static_folder="client/dist", static_url_path="")
root = pw.Config.SCIPION_USER_DATA
root = pathlib.Path(root)


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/user_data")
@app.route("/user_data/<path:subpath>", methods=["GET"])
def user_data(subpath=None):
    subpath = subpath or root
    subpath = root.joinpath(subpath)
    if subpath.is_dir():
        return {"files": [p.name for p in subpath.iterdir()]}
    elif subpath.is_file():
        return send_file(subpath)
    else:
        print(subpath)
        abort(404)


@app.route("/<project>/<ab_initio_run>/dashboard.html")
def ab_initio_dashboard(project, ab_initio_run):
    return app.send_static_file("ab-initio-dashboard.html")
