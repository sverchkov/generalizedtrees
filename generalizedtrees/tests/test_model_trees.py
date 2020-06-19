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

from generalizedtrees.standard_trees import ModelTree
from sklearn.utils.estimator_checks import check_estimator
import pytest


@pytest.mark.skip(reason="can't run this check on modeltree yet")
def test_model_tree_with_sklearn():
    check_estimator(ModelTree)


if __name__ == "__main__":
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=5)

    test_model_tree_with_sklearn()