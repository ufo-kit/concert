import unittest


class TestTango(unittest.TestCase):
    
    def test_tango_dummy(self) -> None:
        self.assertTrue(1 == 1)


if __name__ == "__main__":
    unittest.main()
