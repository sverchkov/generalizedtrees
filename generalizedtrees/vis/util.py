# Utility functions used by visualization components
#
# Licensed under the BSD 3-Clause License
# Copyright (c) 2020, Yuriy Sverchkov

def _ensure_native(value):
    return getattr(value, 'tolist', lambda: value)()