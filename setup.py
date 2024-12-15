from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lattereview",
    version="0.1.0",
    author="Pouria Rouzrokh",
    author_email="po.rouzrokh@gmail.com",
    description="A framework for multi-agent review workflows using large language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PouriaRouzrokh/LatteReview",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
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
        "pandas>=2.2.3",
        "pydantic>=2.10.3",
        "python-dotenv>=1.0.1",
        "tokencost>=0.1.17",
        "tqdm>=4.67.1",
        "openpyxl>=3.1.5",
    ],
    package_data={
        'lattereview': ['generic_prompts/*.txt'],
    },
)