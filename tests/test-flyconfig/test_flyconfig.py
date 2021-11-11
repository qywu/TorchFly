import unittest
from torchfly.flyconfig import FlyConfig 

class TestFlyConfig(unittest.TestCase):
    def test_load_plain_config(self):
        config = FlyConfig.load("config/plain_config.yml")
        self.assertFalse("flylogger" in config)

    def test_load_flylogger_config(self):
        config = FlyConfig.load("config/flylogger_config.yml")
        self.assertTrue("flylogger" in config)
        self.assertFalse("training" in config)

    def test_load_subconfig(self):
        config = FlyConfig.load("config/flylogger_config2.yml")
        self.assertTrue("flylogger" in config)
        self.assertTrue(config.training.random_seed == 123)

if __name__ == '__main__':
    unittest.main()