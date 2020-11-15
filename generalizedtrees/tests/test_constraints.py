# Test constraint classses
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

import pytest
import numpy as np

def test_vector_constraints_tester():

    from generalizedtrees.constraints import test
    from generalizedtrees.constraints import GTConstraint, LEQConstraint

    test_matrix = np.array([
        [0, 5],
        [1, 4],
        [2, 3],
        [3, 2],
        [4, 1],
        [5, 0]
    ])

    np.testing.assert_array_equal(
        test((), test_matrix),
        [True, True, True, True, True, True]
    )

    first_gt_3 = GTConstraint(0, 3)
    first_leq_3 = LEQConstraint(0, 3)
    second_gt_3 = GTConstraint(1, 3)
    second_leq_3 = LEQConstraint(1, 3)

    np.testing.assert_array_equal(
        test((first_gt_3, ), test_matrix),
        [False, False, False, False, True, True]
    )

    np.testing.assert_array_equal(
        test((first_gt_3, second_leq_3), test_matrix),
        [False, False, False, False, True, True]
    )

    np.testing.assert_array_equal(
        test((second_leq_3, first_gt_3), test_matrix),
        [False, False, False, False, True, True]
    )

    np.testing.assert_array_equal(
        test((second_leq_3, first_leq_3), test_matrix),
        [False, False, True, True, False, False]
    )

    np.testing.assert_array_equal(
        test((first_gt_3, second_leq_3, first_leq_3), test_matrix),
        [False, False, False, False, False, False]
    )
