#!/usr/bin/env python3

class IndependenceTest():

    def __init__(self, data, alpha):
        raise NotImplementedError()

    def is_independent(self, x, y, z):
        raise NotImplementedError()

    def is_dependent(self, x, y, z):
        raise NotImplementedError()

    def get_p_value(self):
        raise NotImplementedError()

    def determines(self, z, y):
        raise NotImplementedError()

    def set_alpha(self, alpha):
        raise NotImplementedError()

    def get_alpha(self):
        raise NotImplementedError()
