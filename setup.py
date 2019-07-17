from setuptools import setup, find_packages


def readme():
    with open("README.md", "r") as f:
        return f.read()


setup(
    name="larq-zoo",
    version="0.2.0",
    author="Plumerai",
    author_email="lukas@plumerai.co.uk",
    description="Reference implementations of popular Binarized Neural Networks",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/plumerai/larq-zoo",
    packages=find_packages(),
    license="Apache 2.0",
    install_requires=["numpy~=1.15", "larq~=0.5", "zookeeper~=0.3.1"],
    extras_require={
        "tensorflow": ["tensorflow>=1.13.1"],
        "tensorflow_gpu": ["tensorflow-gpu>=1.13.1"],
        "test": ["pytest>=4.3.1", "pytest-cov>=2.6.1", "Pillow==6.1.0", "scipy==1.3.0"],
    },
    entry_points="""
        [console_scripts]
        lqz=larq_zoo.train:cli
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
