from setuptools import setup, find_packages
from pathlib import Path

setup(
    name="dk154_targets",
    version="0.1.0",
    #description="Run suite for DXS reduction",
    #url="https://github.com/aidansedgewick/dxs",
    author="aidan-sedgewick",
    author_email='aidansedgewick@gmail.com',
    license="MIT license",
    #install_requires=requirements,
    packages = find_packages(),
)

from dk154_targets.paths import create_all_paths
from dk154_targets.utils import dustmaps_config

create_all_paths()
dustmaps_config()

print("are we ready to go?")
