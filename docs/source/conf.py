# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import subprocess


project = 'spfluo-app'
copyright = '2023, Jean Plumail'
author = 'Jean Plumail'
release = subprocess.check_output(["hatch", "version"]).decode()

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinxcontrib.video",
    "sphinx.ext.extlinks",
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_css_files = ['css/custom.css']


numfig = True

# -- Options for PDF output --
latex_engine = 'xelatex'

# extlinks
extlinks = {
    "spfluo-latest-released-files": (f"https://github.com/jplumail/spfluo-app/releases/download/v{release}/%s", "%s"),
    "spfluo-latest-release-page": (f"https://github.com/jplumail/spfluo-app/releases/tag/v{release}%s", None)
}