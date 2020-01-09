# Definitions of splitting strategies for tree-building
#
# Copyright 2020 Yuriy Sverchkov
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

import logging

from generalizedtrees.constraints import LEQConstraint, GTConstraint, EQConstraint, NEQConstraint

logger = logging.getLogger()


def binary_threshold(data, feature):
    for x_i in data:
        yield [LEQConstraint(feature, x_i[feature]), GTConstraint(feature, x_i[feature])]


def one_vs_all(data, feature):
    for x_i in data:
        yield [EQConstraint(feature, x_i[feature]), NEQConstraint(feature, x_i[feature])]


def all_values_split(feature, values):
    yield [EQConstraint(feature, value) for value in values]


def compose_splitting_strategies(spec_list):
    """
    Translates a list of specification into a working splitting strategy function
    :param spec_list:
    :return:
    """
    raise NotImplementedError
