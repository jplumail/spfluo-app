"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name="scipion-em-spfluo",  # Required
    version="0.1",  # Required
    description="Scipion plugin spfluo.",  # Required
    long_description=long_description,  # Optional
    url="https://github.com/scipion-em/scipion-em-spfluo",  # Optional
    author="you",  # Optional
    author_email="jplumail@unistra.fr",  # Optional
    keywords="scipion cryoem imageprocessing scipion-3.0",  # Optional
    packages=find_packages(),
    install_requires=[requirements],
    entry_points={"pyworkflow.plugin": "spfluo = spfluo"},
    package_data={  # Optional
        "spfluo": ["icon.png", "protocols.conf"],
    },
    extra_require={"dev": ["black", "pre-commit"]},
)
