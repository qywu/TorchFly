import os
import time
import shutil
import unittest
import logging

from torchfly.flyconfig import FlyConfig
from torchfly.flylogger import FlyLogger

logger = logging.getLogger(__name__)

config = FlyConfig.load("config/flylogger_config.yml")


class TestFlyConfig(unittest.TestCase):

    def test_case1(self):
        with FlyLogger(config.flylogger, logdir="hello") as flylogger:
            logger.info("qwertyui")
            with open("hello.txt", "w") as f:
                pass

        time.sleep(0.1)

        self.assertTrue(os.path.exists("hello/main.log"))
        

        with open("hello/main.log") as f:
            text = f.read()

        self.assertTrue("qwertyui" in text)
        self.assertTrue(os.path.exists("hello/hello.txt"))
        time.sleep(0.1)

        shutil.rmtree("hello")
        time.sleep(0.1)

    def test_multiple_loggers(self):
        with FlyLogger(config.flylogger, logdir="world", filename="file1.log") as flylogger:
            logger.info("qwertyui")
            with open("hello.txt", "w") as f:
                pass

        time.sleep(0.1)
        self.assertTrue(os.path.exists("world/file1.log"))

        with open("world/file1.log") as f:
            text = f.read()
        self.assertTrue("qwertyui" in text)

        with FlyLogger(config.flylogger, logdir="world2", filename="file2.log") as flylogger:
            logger.info("asdfghjk")
            with open("world.txt", "w") as f:
                pass
        time.sleep(0.1)

        self.assertTrue(os.path.exists("world2/file2.log"))

        with open("world2/file2.log") as f:
            text = f.read()

        self.assertTrue("asdfghjk" in text)

        shutil.rmtree("world")
        shutil.rmtree("world2")
        time.sleep(0.1)


if __name__ == '__main__':
    unittest.main()