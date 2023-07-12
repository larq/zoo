from setuptools import find_packages, setup


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="larq-zoo",
    version="2.3.2",
    author="Plumerai",
    author_email="opensource@plumerai.com",
    description="Reference implementations of popular Binarized Neural Networks",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/plumerai/larq-zoo",
    packages=find_packages(),
    license="Apache 2.0",
    install_requires=[
        "numpy>=1.15.0",
        "packaging>=19",
        "larq>=0.9.2,<0.13.2",
        "zookeeper>=1.0.0",
        "typeguard<3.0.0",
        "protobuf<3.20",
        "importlib-metadata ~= 2.0 ; python_version<'3.8'",
    ],
    extras_require={
        "tensorflow": ["tensorflow>=2.4.0"],
        "tensorflow_gpu": ["tensorflow-gpu>=2.4.0"],
        "test": [
            "black==23.7.0",
            "dill==0.3.6",
            "flake8==6.0.0",
            "isort==5.12.0",
            "pytype==2023.6.16",
            "pytest==7.4.0",
            "pytest-cov==4.1.0",
            "pytest-mock==3.11.1",
            "pytest-xdist==3.3.1",
            "Pillow==10.0.0",
            "tensorflow_datasets>=3.1.0,<4.9.0",
        ],
    },
    entry_points="""
        [console_scripts]
        lqz=larq_zoo.training.main:cli
    """,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
    ],
)
