# Tests for born-again trees
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

def test_tll3(breast_cancer_data_pandas, breast_cancer_rf_model, caplog):

    from generalizedtrees.recipes import TLL3
    from generalizedtrees.vis import explanation_to_JSON
    import pandas as pd
    from time import perf_counter
    import logging

    logger = logging.getLogger()
    caplog.set_level(logging.DEBUG)

    x_train = breast_cancer_data_pandas.x_train
    x_test = breast_cancer_data_pandas.x_test
    model = breast_cancer_rf_model
    target_names = breast_cancer_data_pandas.target_names

    # Verify output shape of model
    logger.debug(f'Model probability prediction:\n{model.predict_proba(x_test)}')   

    # Learn explanation
    t1 = perf_counter()

    logger.info('Creating class instance')
    explain = TLL3()

    logger.info('Fitting tree')
    oracle = lambda x: pd.DataFrame(model.predict_proba(x), columns=target_names)
    explain.fit(x_train, oracle)

    t2 = perf_counter()

    logger.info(f'Time taken: {t2-t1}')

    # Test simple printing
    logger.info(f'Learned tree:\n{explain.show_tree()}')

    # Test JSON
    logger.info(f'JSON: {explanation_to_JSON(explain)}')

    # Make predictions
    logger.info('Running prediction')
    explainer_predictions = explain.predict(x_test)

    logger.info(f'Predictions: {list(explainer_predictions)}')

    logger.info("Done")