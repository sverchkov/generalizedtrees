# Package setup file.
#
# Copyright 2019 Yuriy Sverchkov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="generalizedtrees",
    version="0.0.3",
    author="Yuriy Sverchkov",
    author_email="yuriy.sverchkov@wisc.edu",
    description="Library for tree models: decision trees, model trees, mimic models, etc.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache 2",
    keywords="machine learning decision trees model",
    url="https://github.com/Craven-Biostat-Lab/generalizedtrees",
    packages=['generalizedtrees'],
    install_requires=[
        'scikit-learn',
        'numpy',
        'scipy'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Development Status :: 1 - Planning",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    python_requires='>=3.6'
)
