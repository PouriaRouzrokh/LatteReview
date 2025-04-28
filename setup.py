from setuptools import setup, find_packages


def get_version():
    version = {}
    with open("lattereview/_version.py", "r") as fh:
        exec(fh.read(), version)
    return version["__version__"]


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lattereview",
    version="1.0.6",
    author="Pouria Rouzrokh",
    author_email="po.rouzrokh@gmail.com",
    description="A framework for multi-agent review workflows using large language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PouriaRouzrokh/LatteReview",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "litellm>=1.55.2",
        "nest-asyncio>=1.6.0",
        "ollama>=0.4.4",
        "openai>=1.57.4",
        "pandas>=2.2.2",
        "pydantic>=2.10.3",
        "python-dotenv>=1.0.1",
        "tokencost>=0.1.17",
        "tqdm>=4.67.1",
    ],
    extras_require={
        "dev": [
            "black>=24.10.0",
            "flake8>=7.1.1",
            "ipykernel>=6.29.5",
            "openpyxl>=3.1.5",
            "pillow>=11.0.0",
            "matplotlib>=3.9.4",
            "networkx>=3.2.1",
            "pyvis>=0.3.2",
            "scikit-learn>=1.6.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.0.0",
            "mkdocstrings>=0.24.0",
            "mkdocstrings-python>=1.7.0",
            "mkdocs-gen-files>=0.5.0",
            "mkdocs-literate-nav>=0.6.0",
        ],
        "all": [
            "black>=24.10.0",
            "flake8>=7.1.1",
            "ipykernel>=6.29.5",
            "openpyxl>=3.1.5",
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.0.0",
            "mkdocstrings>=0.24.0",
            "mkdocstrings-python>=1.7.0",
            "mkdocs-gen-files>=0.5.0",
            "mkdocs-literate-nav>=0.6.0",
            "pillow>=11.0.0",
            "matplotlib>=3.9.4",
            "networkx>=3.2.1",
            "pyvis>=0.3.2",
            "scikit-learn>=1.6.0",
        ],
    },
    keywords="review, workflow, machine learning, AI, RIS, literature, systematic review, multi-agent, review workflow, review framework, abstract review, title review, review workflow, review framework, abstract review, title review",
)
