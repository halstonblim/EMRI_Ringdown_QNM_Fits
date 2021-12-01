from setuptools import setup

with open("README.rst", "r") as fh:
    long_description = fh.read()

setup(
    name="fitrd",
    version="0.1.0",
    packages=["fitrd",],
    install_requires=['numpy','pandas','qnm'],
    license="MIT",
    author="Halston Lim",
    author_email="hblim@mit.edu",
    description="Code to fit quasi-normal modes to EMRI waveforms",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    test_suite="nose.collector",
    tests_require=["nose"],
)