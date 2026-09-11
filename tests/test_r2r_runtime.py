import ast
import os
import unittest

from vlnce_baselines.common.r2r_runtime import (
    initialize_runtime_config,
    prepare_r2r_evaluation_environment,
)
from vlnce_baselines.config.default import get_config


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class R2RRuntimeWiringTest(unittest.TestCase):
    def test_runtime_config_reaches_formal_eval_sensor_setup_without_trainer(self):
        config = get_config(
            "run_r2r/iter_train.yaml",
            [
                "SIMULATOR_GPU_IDS", "[0]", "TORCH_GPU_IDS", "[0]",
                "TORCH_GPU_ID", "0", "GPU_NUMBERS", "1",
                "NUM_ENVIRONMENTS", "1", "EVAL.SPLIT", "val_unseen",
                "EVAL.EPISODE_IDS", "['53']",
            ],
        )
        initialize_runtime_config(config, local_rank=0)
        prepared = prepare_r2r_evaluation_environment(config)

        self.assertTrue(prepared.is_frozen())
        self.assertEqual(prepared.local_rank, 0)
        self.assertEqual(prepared.SIMULATOR_GPU_IDS, [0])
        self.assertEqual(prepared.TASK_CONFIG.DATASET.SPLIT, "val_unseen")
        self.assertFalse(prepared.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE)
        self.assertIn("RGB_30", prepared.SENSORS)
        self.assertIn("DEPTH_30", prepared.SENSORS)

    def test_run_exp_and_sensor_smoke_use_the_shared_runtime_initializer(self):
        with open(os.path.join(REPO_ROOT, "run.py")) as handle:
            run_tree = ast.parse(handle.read())
        with open(os.path.join(REPO_ROOT, "scripts", "mp3d_episode_sensor_smoke.py")) as handle:
            smoke_tree = ast.parse(handle.read())

        def calls(tree):
            return {
                node.func.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }

        self.assertIn("initialize_runtime_config", calls(run_tree))
        self.assertIn("initialize_runtime_config", calls(smoke_tree))
        self.assertIn("prepare_r2r_evaluation_environment", calls(smoke_tree))

    def test_formal_evaluator_uses_the_sensor_setup_shared_with_smoke(self):
        path = os.path.join(REPO_ROOT, "vlnce_baselines", "common", "base_il_trainer.py")
        with open(path) as handle:
            tree = ast.parse(handle.read())
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("prepare_r2r_evaluation_environment", calls)

    def test_runner_preserves_incomplete_attempts_and_refuses_completed_evidence(self):
        path = os.path.join(REPO_ROOT, "scripts", "run_original_forensics_manual.bash")
        with open(path) as handle:
            runner = handle.read()
        self.assertIn('"${output_root}/COMPLETE"', runner)
        self.assertIn('"${output_root}/FAILED"', runner)
        self.assertIn('.incomplete.$(date +%Y%m%d-%H%M%S-%N)', runner)
        self.assertIn('mv "${output_root}" "${failed_root}"', runner)


if __name__ == "__main__":
    unittest.main()
