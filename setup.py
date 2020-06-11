from setuptools import find_packages, setup


def readme():
    with open("README.md", "r") as f:
        return f.read()


setup(
    name="larq-zoo",
    version="1.0.3",
    author="Plumerai",
    author_email="lukas@plumerai.co.uk",
    description="Reference implementations of popular Binarized Neural Networks",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/plumerai/larq-zoo",
    packages=find_packages(),
    license="Apache 2.0",
    install_requires=["numpy~=1.15", "larq>=0.9.2,<0.10.0", "zookeeper~=1.0.0"],
    extras_require={
        "tensorflow": ["tensorflow>=1.15.0"],
        "tensorflow_gpu": ["tensorflow-gpu>=1.15.0"],
        "test": [
            "black==19.10b0",
            "flake8>=3.7.9,<3.9.0",
            "isort~=4.3.21",
            "pytype>=2019.10.17,<2020.7.0",
            "pytest>=4.3.1",
            "pytest-cov>=2.6.1",
            "pytest-xdist==1.32.0",
            "Pillow==7.1.2",
            "scipy==1.4.1",
            "tensorflow_datasets>=3.1.0",
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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
    ],
)
