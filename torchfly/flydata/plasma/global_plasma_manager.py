import os
import time
import subprocess
import psutil
import shutil
import atexit
import pyarrow as pa
import pyarrow.plasma as plasma
from omegaconf import OmegaConf
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class Singleton(type):
    """A metaclass that creates a Singleton base class when called."""
    _instances: Dict[type, "Singleton"] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class GlobalPlasmaManager(metaclass=Singleton):
    def __init__(self, config: OmegaConf = None):
        """Initialize a Plasma object."""
        if config is not None:
            self.plasma_store_name = config.plasma_store_name
            self.use_mem_percent = config.use_mem_percent
        else:
            self.plasma_store_name = "plasma"
            self.use_mem_percent = 0.30

        self.connected = False
        self.plasma_store_address = None
        self.plasma_store_path = None
        self.plasma_store_proc = None
        self.client = None

        self.initialize(self.plasma_store_name)

        assert self.client is not None

        # shut down plasma when the program is killed
        atexit.register(self.__del__)

    def initialize(self, plasma_store_name):
        """
        os.environ["LOCAL_RANK"] is checked to make sure there is only one Plasma Store Server is runing on each local machine 
        """
        if self.connected:
            raise ValueError("Plasma has already been initialized!")

        if "LOCAL_RANK" not in os.environ or os.environ["LOCAL_RANK"] == 0:
            memory = psutil.virtual_memory()
            plasma_store_memory = int(memory.available * self.use_mem_percent)

            self.plasma_store_path, self.plasma_store_proc = _start_plasma_store(
                plasma_store_memory, plasma_store_name
            )

            self.connected = True
            logger.info(
                f"Initializing Plasma with {plasma_store_memory // 1e9} GB Memory\n"
                f"    Plasma Location on {plasma_store_name}"
            )
            self.client = plasma.connect(self.plasma_store_path)
        else:
            # assume plasma server is running
            self.plasma_store_path = f"/tmp/torchfly/{plasma_store_name}-{hash(os.getcwd())}/plasma.sock"
            local_rank = os.environ["LOCAL_RANK"]
            logger.info(f"Plasma Store on {local_rank} is connected!")
            self.client = plasma.connect(self.plasma_store_path)

        logger.info("Plasma Client Connected!")

    def is_connected(self) -> bool:
        return self.connected

    def __repr__(self):
        return (f"Plasma Store on: \n" f"    Addr: {self.plasma_store_address}\n" f"    Path: {self.plasma_store_path}")

    def __del__(self):
        "Destructor of PlasmaManager"
        if self.plasma_store_path and os.path.exists(self.plasma_store_path):
            shutil.rmtree(os.path.dirname(self.plasma_store_path))

        if isinstance(self.plasma_store_proc, subprocess.Popen):
            self.plasma_store_proc.kill()

        if self.connected:
            logger.info("Plasma is ended!")
            self.connected = False


def _start_plasma_store(
    plasma_store_memory: int,
    plasma_store_name: str = "plasma",
    use_valgrind: bool = False,
    use_profiler: bool = False,
    use_hugepages: bool = False,
):
    """Start a plasma store process.
    Args:
        plasma_store_memory (int): Capacity of the plasma store in bytes.
        use_valgrind (bool): True if the plasma store should be started inside
            of valgrind. If this is True, use_profiler must be False.
        use_profiler (bool): True if the plasma store should be started inside
            a profiler. If this is True, use_valgrind must be False.
        use_hugepages (bool): True if the plasma store should use huge pages.
    Return:
        A tuple of the name of the plasma store socket and the process ID of
            the plasma store process.
    """

    if use_valgrind and use_profiler:
        raise Exception("Cannot use valgrind and profiler at the same time.")

    # datetime.datetime.now().strftime("torchfly/session_%Y-%m-%d_%H-%M-%S_%s"
    stamp = f"/tmp/torchfly/{plasma_store_name}-{hash(os.getcwd())}"
    os.makedirs(stamp, exist_ok=True)

    plasma_store_path = os.path.join(stamp, 'plasma.sock')
    plasma_store_executable = os.path.join(pa.__path__[0], "plasma-store-server")
    command = [plasma_store_executable, "-s", plasma_store_path, "-m", str(plasma_store_memory)]

    if use_hugepages:
        command += ["-h"]

    stdout_file = None
    stderr_file = None
    if use_valgrind:
        command = [
            "valgrind", "--track-origins=yes", "--leak-check=full", "--show-leak-kinds=all",
            "--leak-check-heuristics=stdstring", "--error-exitcode=1"
        ] + command
        proc = subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file)
        time.sleep(1.0)
    elif use_profiler:
        command = ["valgrind", "--tool=callgrind"] + command
        proc = subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file)
        time.sleep(1.0)
    else:
        proc = subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file)
        time.sleep(0.1)
    rc = proc.poll()
    if rc is not None:
        raise RuntimeError("plasma_store exited unexpectedly with " "code %d" % (rc, ))

    return plasma_store_path, proc


# def get_plasma_manager() -> PlasmaManager:
#     if not is_plasma_initialized():
#         raise RuntimeError("Please call `init_plasma` in the main process first before using plasma!")
#     return _global_manager

# if __name__ == "__main__":
#     # test
#     init_plasma()
#     client = plasma.connect(_global_manager.plasma_store_address)
#     client.put("xxx")
#     print("Stopped")