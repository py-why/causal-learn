#!/usr/bin/env python3
import unittest

from data.DataSet import DataSet
from data.DataUtils import DataUtils

class TestDataMethods(unittest.TestCase):

    # def test_continuous_dataloader(self):
    #
    #     utils = DataUtils()
    #
    #     dataset = utils.load_continuous_data('tests/test_data_set.txt')
    #
    #     print(str(dataset))

    # def test_discrete_dataloader(self):
    #
    #     utils = DataUtils()
    #
    #     dataset = utils.load_discrete_data('tests/test_data_set.txt')
    #
    #     print(str(dataset))

    def test_mixed_dataloader(self):

        utils = DataUtils()

        dataset = utils.load_mixed_data('tests/test_data_set_3.txt', 3)

        print(str(dataset))


if __name__ == '__main__':
    unittest.main()
