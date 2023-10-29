__author__ = "bgravell"

from setuptools import setup, find_packages

setup(
    name="terra",
    version="0.0.2",
    python_requires=">=3.11",
    author="Benjamin Gravell",
    author_email="",
    description="Country recommendation streamlit app",
    license="MIT",
    packages=find_packages(),
    package_data={"terra": ["assets/*", "data/*", "help/*"]},
    include_package_data=True,
    long_description=open("README.md").read(),
)
