

from string import Template
from generalizedtrees.vis.vis import _load_html_template, explanation_to_JSON, _load_dt_js

def draw(explanation, feature_annotations = None):

    return Template(_load_html_template('nb_template.html')).substitute({
        'data': explanation_to_JSON(explanation, feature_annotations),
        'artist': _load_dt_js()
    })