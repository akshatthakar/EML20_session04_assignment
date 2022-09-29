#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="1.0.1",
    description="Assignment for EML V2 Session 2",
    author="akshat",
    author_email="akshat.thakar@gmail.com",
    url="https://github.com/DataInsightMLOps/EML20_session02_assignment.git",  
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
)
