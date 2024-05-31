import os
from setuptools import setup, find_packages

def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname), "rb") as f:
        reqs = f.read().decode("utf-8")
    return reqs

#packages = find_packages()
packages = [
    "DeepUrfold",
    "DeepUrfold.Analysis",
    "DeepUrfold.Analysis.AllVsAll",
    "DeepUrfold.Analysis.AllVsAll.SequenceBased",
    "DeepUrfold.Analysis.AllVsAll.StructureBased",
    "DeepUrfold.Analysis.Webapp",
    "DeepUrfold.DataModules",
    "DeepUrfold.Models",
    "DeepUrfold.Metrics",
    "DeepUrfold.Trainers",
    "DeepUrfold.Evaluators",
    "DeepUrfold.util",
    "DeepUrfold.scripts"
]


#find_packages(exclude=("binding_site_test",))
print(packages)

setup(
    name = "DeepUrfold",
    version = "0.0.1",
    author = "Eli Draizen",
    author_email = "edraizen@gmail.com",
    packages=packages,
    long_description=read('README.md'),
    install_requires=read("requirements.txt"),
    include_package_data=True
)
