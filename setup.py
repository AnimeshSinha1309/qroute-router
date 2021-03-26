import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="qroute",
    version="v0.1.0",
    author="Animesh Sinha",
    author_email="animesh.sinha@research.iiit.ac.in",
    description="A library for Machine Learning algorithms in Qubit Routing.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AnimeshSinha1309/quantum-rl/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
