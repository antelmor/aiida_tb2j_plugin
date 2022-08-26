import codecs
import json
import os
from setuptools import find_packages, setup

THIS_LOC = os.path.abspath(os.path.dirname(__file__))

__version__ = "0.1.dev1"

def read(*parts):
    with codecs.open(os.path.join(THIS_LOC, *parts), "rb", "utf-8") as filenam:
        return filenam.read()


if __name__ == '__main__':
    with open('setup.json', 'r') as info:
        kwargs = json.load(info)  # pylint: disable=invalid-name
    setup(
        include_package_data=True,
        packages=find_packages(),
        reentry_register=True,
        **kwargs
    )
