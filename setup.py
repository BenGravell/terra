__author__ = "bgravell"

from setuptools import setup

setup(
    name="terra",
    version="0.0.1",
    author="Benjamin Gravell",
    author_email="",
    description="Country recommendation streamlit app",
    license="MIT",
    packages=["culture_fit", "culture_fit.country_data"],
    long_description=open("README.md").read(),
    include_package_data=False,
    zip_safe=False,
)
