"""Package SMART and its vendored LAVIS modules."""
from pathlib import Path
from setuptools import find_namespace_packages, setup

ROOT = Path(__file__).resolve().parent
requirements = [
    line.split("#", 1)[0].strip()
    for line in (ROOT / "requirements.txt").read_text().splitlines()
    if line.split("#", 1)[0].strip()
]
setup(
    name="smart-moment-retrival",
    version="0.1.0",
    description="SMART: Shot-Aware Multimodal Video Moment Retrieval with Audio-Enhanced MLLM",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license="BSD-3-Clause",
    packages=find_namespace_packages(include=["lavis_1019", "lavis_1019.*"]),
    install_requires=requirements,
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=False,
)
