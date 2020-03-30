import os
import time
import datetime
import subprocess
import pyarrow as pa
import pyarrow.plasma as plasma
import psutil
import shutil
import tempfile
import atexit
import logging

logger = logging.getLogger(__name__)


class PlasmaManager:
    def __init__(self):
        """Initialize a Manager object."""
        self.connected = False
        self.plasma_store_address = None
        self.plasma_store_dir = None
        self.plasma_store_proc = None
        self._session_index = 0

    def __del__(self):
        "Destructor of PlasmaManager"
        if self.plasma_store_dir and os.path.exists(self.plasma_store_dir):
            shutil.rmtree(self.plasma_store_dir)

        if isinstance(self.plasma_store_proc, subprocess.Popen):
            self.plasma_store_proc.kill()

        logger.info("Plasma is ended!")


_global_manager = PlasmaManager()
atexit.register(_global_manager.__del__)


def init_plasma(plasma_store_memory: int = None):
    global _global_manager

    memory = psutil.virtual_memory()
    plasma_store_memory = int(memory.available * 0.3)
    plasma_store_name, plasma_dir, process = _start_plasma_store(plasma_store_memory)
    _global_manager.plasma_store_address = plasma_store_name
    _global_manager.plasma_store_dir = plasma_dir
    _global_manager.plasma_store_proc = process
    _global_manager.connected = True

    logger.info(
        f"Initialize Plasma with {plasma_store_memory // 1e9} GB Memory\nPlasma Location on {plasma_store_name}"
    )

    return _global_manager


def is_plasma_initialized() -> bool:
    return _global_manager.connected


def get_plasma_manager() -> PlasmaManager:
    if not is_plasma_initialized():
        raise RuntimeError("Please call `init_plasma` in the main process first before using plasma!") 
    return _global_manager


def _start_plasma_store(
    plasma_store_memory,
    use_valgrind=False,
    use_profiler=False,
    plasma_directory=None,
    use_hugepages=False,
    external_store=None
):
    """Start a plasma store process.
    Args:
        plasma_store_memory (int): Capacity of the plasma store in bytes.
        use_valgrind (bool): True if the plasma store should be started inside
            of valgrind. If this is True, use_profiler must be False.
        use_profiler (bool): True if the plasma store should be started inside
            a profiler. If this is True, use_valgrind must be False.
        plasma_directory (str): Directory where plasma memory mapped files
            will be stored.
        use_hugepages (bool): True if the plasma store should use huge pages.
        external_store (str): External store to use for evicted objects.
    Return:
        A tuple of the name of the plasma store socket and the process ID of
            the plasma store process.
    """

    if use_valgrind and use_profiler:
        raise Exception("Cannot use valgrind and profiler at the same time.")

    stamp = datetime.datetime.now().strftime("torchfly/session_%Y-%m-%d_%H-%M-%S_%s")
    os.makedirs("/tmp/torchfly", exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix=f'{stamp}-')

    plasma_store_name = os.path.join(tmpdir, 'plasma.sock')
    plasma_store_executable = os.path.join(pa.__path__[0], "plasma-store-server")
    command = [plasma_store_executable, "-s", plasma_store_name, "-m", str(plasma_store_memory)]
    
    if use_hugepages:
        command += ["-h"]
    if external_store is not None:
        command += ["-e", external_store]
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

    return plasma_store_name, tmpdir, proc


if __name__ == "__main__":
    # test
    init_plasma()
    client = plasma.connect(_global_manager.plasma_store_address)
    client.put("xxx")
    print("Stopped")