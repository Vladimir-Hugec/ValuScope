from setuptools import setup, find_packages

setup(
    name="valuscope",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "yfinance>=0.1.63",
        "seaborn>=0.11.0",
        "pandas-datareader>=0.10.0",
    ],
    entry_points={
        "console_scripts": [
            "valuscope=valuscope.main:main",
        ],
    },
    author="Vladimir Hugec",
    author_email="vladimir.hugec.jr@gmail.com",
    description="ValuScope: A toolkit for financial analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Vladimir-Hugec/ValuScope",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.10",
)
