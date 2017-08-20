# /setup.py
#
# Installation and setup script for Main Roads Cloud Vision
#
# See /LICENCE.md for Copyright information
"""Installation and setup script for Main Roads Cloud Vision."""

from setuptools import find_packages, setup

setup(name="tensorflow-recognizer-cli",
      version="0.0.1",
      description="""Tensorflow Recognizer.""",
      long_description=(
          """Command line interface to recognize images with tensorflow."""
      ),
      author="Sam Spilsbury",
      author_email="smspillaz@gmail.com",
      classifiers=["Development Status :: 3 - Alpha",
                   "Programming Language :: Python :: 2",
                   "Programming Language :: Python :: 2.7",
                   "Programming Language :: Python :: 3",
                   "Programming Language :: Python :: 3.1",
                   "Programming Language :: Python :: 3.2",
                   "Programming Language :: Python :: 3.3",
                   "Programming Language :: Python :: 3.4",
                   "Intended Audience :: Developers",
                   "Topic :: Software Development :: Build Tools",
                   "License :: OSI Approved :: MIT License"],
      url="http://github.com/smspillaz/main-roads-cloud-vision",
      license="ISC",
      keywords="nlp",
      packages=find_packages(
          exclude=["build", "dist", "*.egg-info", "*node_modules*"]
      ),
      install_requires=[
          "setuptools"
      ],
      entry_points={
          "console_scripts": [
              "tensorflow-recognize=project.tensorflow:main"
          ]
      },
      zip_safe=True,
      include_package_data=True)
