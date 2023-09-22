"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# To use a consistent encoding
from codecs import open
from os import path

# Always prefer setuptools over distutils
from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name="scipion-fluo-singleparticle",  # Required
    description="Scipion plugin for single particle reconstruction.",  # Required
    long_description=long_description,  # Optional
    url="https://github.com/jplumail/scipion-fluo-singleparticle",  # Optional
    author="you",  # Optional
    author_email="jplumail@unistra.fr",  # Optional
    keywords="scipion fluorescence imageprocessing scipion-3.0",  # Optional
    packages=find_packages(),
    install_requires=[requirements],
    entry_points={"pyworkflow.plugin": "singleparticle = singleparticle"},
    package_data={  # Optional
        "singleparticle": ["icon.png", "protocols.conf"],
    },
    extra_require={"dev": ["black", "pre-commit", "ruff"]},
)
