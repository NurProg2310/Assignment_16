import unittest

import hw
class TestStaticArray(unittest.TestCase):

    def setUp(self):
        self.array = hw.StaticArray(5)

    def test_set_get(self):
        self.array.set(0, 5)
        self.assertEqual(self.array.get(0), 5)

        self.array.set(1, 10)
        self.assertEqual(self.array.get(1), 10)

        self.array.set(4, 20)
        self.assertEqual(self.array.get(4), 20)
class TestDynamicArray(unittest.TestCase):

    def setUp(self):
        self.array = hw.DynamicArray()

    def test_append(self):
        self.array.append(5)
        self.assertEqual(self.array.get(0), 5)

    def test_insert(self):
        self.array.insert(0, 5)
        self.assertEqual(self.array.get(0), 5)

    def test_delete(self):
        self.array.append(5)
        self.array.append(10)
        self.array.delete(0)
        self.assertEqual(self.array.get(0), 10)