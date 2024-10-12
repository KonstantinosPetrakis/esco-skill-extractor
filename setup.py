from setuptools import setup, find_packages


requirements = [r.strip() for r in open("requirements.txt").readlines()]
nvidia_requirements = [
    r for r in requirements if r.startswith("nvidia") or r.startswith("triton")
]
standard_requirements = [r for r in requirements if r not in nvidia_requirements]

setup(
    name="esco-skill-extractor",
    version="0.1.8",
    packages=find_packages(),
    install_requires=standard_requirements,
    extras_require={"cuda": nvidia_requirements},
    include_package_data=True,
    package_data={"esco_skill_extractor": ["data/*.csv"]},
    author="Konstantinos Petrakis",
    author_email="konstpetrakis01@gmail.com",
    description="Extract ESCO skills from texts such as job descriptions or CVs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/KonstantinosPetrakis/esco-skill-extractor",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
