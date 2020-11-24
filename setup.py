from setuptools import setup, find_packages


PACKAGE = "ad"
NAME = PACKAGE
DESCRIPTION = "Operator overloading based autograd"
AUTHOR = "tor4z"
AUTHOR_EMAIL = "vwenjie@hotmail.com"
URL = "https://github.com/tor4z/AD_OO"
LICENSE = "MIT License"
VERSION = 0.01


setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      url=URL,
      packages=find_packages(exclude=["test"]))
