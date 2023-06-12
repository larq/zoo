from setuptools import find_packages, setup


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="larq-zoo",
    version="2.3.1",
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
        "larq>=0.9.2,<0.14.0",
        "zookeeper>=1.0.0,<1.4.0",
        "importlib-metadata ~= 2.0 ; python_version<'3.8'",
    ],
    extras_require={
        "tensorflow": ["tensorflow>=1.15.0"],
        "tensorflow_gpu": ["tensorflow-gpu>=1.15.0"],
        "test": [
            "black==23.3.0",
            "flake8==5.0.4",
            "isort==5.11.4",
            "pytype>=2022.01.05",
            "pytest==7.3.2",
            "pytest-cov==4.0.0",
            "pytest-mock>=3.6.1,<3.11.0",
            "pytest-xdist==3.2.1",
            "Pillow>=8.4.0,<10.0.0",
            "tensorflow_datasets>=3.1.0,!=4.3.0",
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
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
    ],
)
