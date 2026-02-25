from setuptools import setup, find_packages

setup(
    name="procure-iq",
    version="0.1.0",
    description="Agentic AI + GNN platform for procurement intelligence",
    author="Shruti Dhamdhere",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)
