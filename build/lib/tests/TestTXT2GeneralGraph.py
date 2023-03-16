import sys

sys.path.append("")
import unittest

from causallearn.utils.TXT2GeneralGraph import txt2generalgraph


class TestTXT2GeneralGraph(unittest.TestCase):
    def test_read_from_txt_case1(self):
        for i in range(1, 11):
            G = txt2generalgraph(f"fci-test-data/True-Graph{i}.txt")
            print(G)
