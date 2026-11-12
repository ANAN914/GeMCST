from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
setup(
    name="GeMCST",
    version="1.0.0",
    description="Multimodal spatial transcriptomics domain identification",
    author="hanyiyu",
    license="MIT",
    license_files=["LICENSE"],
    packages=find_packages(include=["GeMCST", "GeMCST.*"]),
    install_requires=(ROOT / "requirements.txt").read_text().splitlines(),
    python_requires=">=3.10",
    entry_points={"console_scripts": ["gemcst=GeMCST.main:main"]},
    package_data={"GeMCST": ["configs/*.json"]},
    include_package_data=True,
    zip_safe=False,
)
