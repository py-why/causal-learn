#!/usr/bin/env python3

from enum import Enum


class NodeType(Enum):
    MEASURED = 1
    LATENT = 2
    ERROR = 3
    SESSION = 4
    RANDOMIZE = 5
    LOCK = 6
    NO_TYPE = 7

    def __str__(self):
        return self.name
