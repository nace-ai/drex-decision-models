"""gpu_lock() must work on an immutable lock file (chflags uchg); CPU only, no model."""
import fcntl
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import drex_max_mlx


@unittest.skipUnless(sys.platform == "darwin", "chflags uchg is BSD/macOS only")
class ImmutableLockTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "gpu.lock")
        Path(self.path).touch()
        subprocess.run(["chflags", "uchg", self.path], check=True)
        env = {k: v for k, v in os.environ.items() if k not in ("NI_LOCK_HELD", "NACE_GPU_LOCK_HELD")}
        env["BENCH_GPU_LOCK"] = self.path
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        subprocess.run(["chflags", "nouchg", self.path], check=True)
        os.unlink(self.path)
        os.rmdir(self.dir)

    def test_takes_lock_on_immutable_file(self):
        with drex_max_mlx.gpu_lock():
            fd = os.open(self.path, os.O_RDONLY)
            try:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(fd)
        self.assertTrue(os.stat(self.path).st_flags & 0x2)  # UF_IMMUTABLE kept

    def test_busy_lock_still_refused(self):
        fd = os.open(self.path, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "GPU is busy"):
                with drex_max_mlx.gpu_lock():
                    pass
        finally:
            os.close(fd)


if __name__ == "__main__":
    unittest.main()
