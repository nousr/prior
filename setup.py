from setuptools import setup, find_packages
from pathlib import Path
import os

if __name__ == "__main__":
    with Path(Path(__file__).parent, "README.md").open(encoding="utf-8") as file:
        long_description = file.read()

    setup(
        name="prior",
        packages=find_packages(),
        include_package_data=True,
        version="0.0.1",
        license="MIT",
        description="prior models for embedding translation",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Zion English",
        author_email="zion.m.english@gmail.com",
        url="https://github.com/nousr/prior",
        data_files=[(".", ["README.md"])],
        keywords=["machine learning", "pytorch"],
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.10",
        ],
    )
