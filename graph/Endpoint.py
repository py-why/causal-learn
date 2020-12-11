#!/usr/bin/env python3
from enum import Enum

# A typesafe enumeration of the types of endpoints that are permitted in
# Tetrad-style graphs: null (-), arrow (->), and circle (-o).
class Endpoint(Enum):

    TAIL = 1
    ARROW = 2
    CIRCLE = 3
    STAR = 4
    NULL = 5

    # Prints out the name of the type
    def __str__(self):
        return self.name
