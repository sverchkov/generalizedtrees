# Utility functions
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

from functools import total_ordering

## Decorator for making a class orderable
def order_by(*attrs):
    """
    Make a class orderable by a set of its attributes.

    Class decorator.

    Parameters are the names of attributes in order of priority.
    This defines all comparison operators (and __eq__) based on the values of the attributes.
    (Actually we only define __lt__ and __eq__ and pass to functools.total_ordering to define the rest.)
    """

    def decorate(cls):

        def eq(self, other):
            return all(getattr(self, a) == getattr(other, a) for a in attrs)
        
        def lt(self, other):

            for a in attrs:
                ours = getattr(self, a)
                theirs = getattr(other, a)
                if ours < theirs: return True
                elif theirs > ours: return False
            
            return False
        
        setattr(cls, '__lt__', lt)
        setattr(cls, '__eq__', eq)

        return total_ordering(cls)
    
    return decorate
