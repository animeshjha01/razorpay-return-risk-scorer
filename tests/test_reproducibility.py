import unittest
import os
import shutil
import filecmp
from src.generate_data import generate_data

class TestReproducibility(unittest.TestCase):
    def setUp(self):
        self.temp_dir_1 = "tests/temp_data_1"
        self.temp_dir_2 = "tests/temp_data_2"
        self.temp_dir_3 = "tests/temp_data_3"
        for d in [self.temp_dir_1, self.temp_dir_2, self.temp_dir_3]:
            os.makedirs(d, exist_ok=True)
            
    def tearDown(self):
        for d in [self.temp_dir_1, self.temp_dir_2, self.temp_dir_3]:
            if os.path.exists(d):
                shutil.rmtree(d)
                
    def _run_generator_to_dir(self, out_dir, seed):
        # We temporarily chdir so the script writes data/train.csv inside out_dir
        orig_cwd = os.getcwd()
        os.chdir(out_dir)
        try:
            generate_data(n=1000, seed=seed, test_size=0.2)
        finally:
            os.chdir(orig_cwd)

    def test_same_seed_is_byte_for_byte_identical(self):
        self._run_generator_to_dir(self.temp_dir_1, 42)
        self._run_generator_to_dir(self.temp_dir_2, 42)
        
        train1 = os.path.join(self.temp_dir_1, 'data', 'train.csv')
        train2 = os.path.join(self.temp_dir_2, 'data', 'train.csv')
        test1 = os.path.join(self.temp_dir_1, 'data', 'test.csv')
        test2 = os.path.join(self.temp_dir_2, 'data', 'test.csv')
        
        self.assertTrue(os.path.exists(train1))
        self.assertTrue(filecmp.cmp(train1, train2, shallow=False), "train.csv files should be byte-for-byte identical")
        self.assertTrue(filecmp.cmp(test1, test2, shallow=False), "test.csv files should be byte-for-byte identical")

    def test_different_seed_is_different(self):
        self._run_generator_to_dir(self.temp_dir_1, 42)
        self._run_generator_to_dir(self.temp_dir_3, 43)
        
        train1 = os.path.join(self.temp_dir_1, 'data', 'train.csv')
        train3 = os.path.join(self.temp_dir_3, 'data', 'train.csv')
        
        self.assertTrue(os.path.exists(train1))
        self.assertTrue(os.path.exists(train3))
        self.assertFalse(filecmp.cmp(train1, train3, shallow=False), "train.csv files should differ when seed differs")

if __name__ == '__main__':
    unittest.main()
