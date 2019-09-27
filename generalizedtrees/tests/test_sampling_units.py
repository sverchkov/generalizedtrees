# Tests for standard trees
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
from generalizedtrees.constraints import LEQConstraint
from generalizedtrees.sampling import rejection_sample_generator
from numpy import random
import pytest


@pytest.mark.slow
def test_constrained_generator():
    rjc = rejection_sample_generator(lambda n: random.uniform(low=0, high=10, size=(n, 4)))
    print(rjc(5, (LEQConstraint(2, 5),)))


if __name__ == "__main__":
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=5)

    test_constrained_generator()

