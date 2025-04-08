import os
import setuptools

# Read the contents of the README file
with open(os.path.join(os.path.dirname(__file__), "README.md"), "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wizx-sdk",
    version="1.0.0",
    author="WIZX Agricultural Platform",
    author_email="api@wizx.io",
    description="SDK for the WIZX Agricultural Commodity Pricing Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://wizx.io",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        "requests>=2.25.0",
    ],
)