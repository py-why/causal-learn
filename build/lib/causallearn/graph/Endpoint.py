#!/usr/bin/env python3
from enum import Enum


# A typesafe enumeration of the types of endpoints that are permitted in
# Tetrad-style graphs: tail (--) null (-), arrow (->), circle (-o) and star (-*).
# 'TAIL_AND_ARROW' and 'ARROW_AND_ARROW' means there are two types of edges (<-> and -->)
# between two nodes
class Endpoint(Enum):
    TAIL = -1
    NULL = 0
    ARROW = 1
    CIRCLE = 2
    STAR = 3
    TAIL_AND_ARROW = 4
    ARROW_AND_ARROW = 5

    # Prints out the name of the type
    def __str__(self):
        return self.name
