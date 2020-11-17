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

import generalizedtrees
from setuptools import setup, find_packages

with open("README.rst", "r") as fh:
    long_description = fh.read()

with open("oldest") as fh:
    requirements = fh.read().replace("==", ">=")

setup(
    name="generalizedtrees",
    version=generalizedtrees.__version__,
    author="Yuriy Sverchkov",
    author_email="yuriy.sverchkov@wisc.edu",
    description="Library for tree models: decision trees, model trees, mimic models, etc.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    license="BSD (3 clause)",
    keywords="machine learning decision trees model",
    url="https://github.com/Craven-Biostat-Lab/generalizedtrees",
    packages=find_packages(exclude=['*.tests']),
    package_data={'generalizedtrees.vis': ['*.html', '*.js']},
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Development Status :: 2 - Pre-Alpha",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    python_requires='>=3.8'
)
