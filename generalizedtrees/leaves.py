# Leaf models for trees
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

from numpy import ndarray, full


class SimplePredictor:

    def __init__(self, prediction):
        self.prediction = prediction

    def predict_one(self, sample):
        return self.prediction

    def predict(self, data: ndarray):
        assert data.ndim == 2
        return full(data.shape[0], self.prediction)

    def __repr__(self):
        return f'Predict: {self.prediction}'


class ClassificationPredictor:

    def __init__(self, classifier):
        self._classifier = classifier

    def predict_one(self, sample):
        return self._classifier.predict_one(sample.reshape((1, -1)))[0]

    def __repr__(self):
        return f'Predict: {self._classifier}'
