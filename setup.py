import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="prz-3m-gustavozf",
    version="0.0.1",
    author="Gustavo Zanoni Felipe",
    author_email="gzf1996@gmail.com",
    description="A Patter Recognition library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gustavozf/PRZ_3M",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires='>=3.6',
)
