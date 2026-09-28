import os
import unittest
from unittest.mock import patch

from compute_guard import require_compute_step


class ComputeGuardTests(unittest.TestCase):
    def test_no_allocation_rejected(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(SystemExit):
            require_compute_step()

    def test_inherited_job_on_login_rejected(self):
        env = {"SLURM_JOB_ID": "123", "SLURMD_NODENAME": "compute1", "SLURM_CPUS_PER_TASK": "2"}
        with patch.dict(os.environ, env, clear=True), patch("socket.gethostname", return_value="login1"), self.assertRaises(SystemExit):
            require_compute_step()

    def test_allocated_node_enforces_threads(self):
        env = {"SLURM_JOB_ID": "123", "SLURMD_NODENAME": "compute1", "SLURM_CPUS_PER_TASK": "2"}
        with patch.dict(os.environ, env, clear=True), patch("socket.gethostname", return_value="compute1.local"):
            require_compute_step(2)
            self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "1")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
            with self.assertRaises(SystemExit):
                require_compute_step(3)

    def test_missing_cpu_request_rejected(self):
        env = {"SLURM_JOB_ID": "123", "SLURMD_NODENAME": "compute1"}
        with patch.dict(os.environ, env, clear=True), patch("socket.gethostname", return_value="compute1"), self.assertRaises(SystemExit):
            require_compute_step()


if __name__ == "__main__":
    unittest.main()
