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

create_all_paths()

print("configure dustmaps")
from dustmaps.config import config
if not Path(config.fname).exists():
    config.reset()
from dustmaps import sfd
try:
    sfdq = sfd.query()
except:
    sfd.fetch()

print("are we ready to go?")
