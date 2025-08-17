from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cardioinsight-ai",
    version="1.0.0",
    author="Ahmed Umer Farooq",
    author_email="ahmedumerfarooq@example.com",
    description="A professional machine learning application for cardiovascular risk assessment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ahmed-Umer-Farooq/Heart-Disease-Detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "cardioinsight=app:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)