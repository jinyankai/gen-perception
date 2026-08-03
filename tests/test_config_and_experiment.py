import tempfile
import unittest
from pathlib import Path

from perception_diffusion.utils.config import ConfigError, load_config, validate_config
from perception_diffusion.utils.experiment import create_experiment_directory


ROOT = Path(__file__).resolve().parents[1]


class ConfigTest(unittest.TestCase):
    def test_smoke_config_loads(self):
        config = load_config(ROOT / "configs" / "smoke.yaml", expand_environment=False)
        self.assertEqual("segmentation", config["task"]["name"])

    def test_invalid_task_is_rejected(self):
        config = {
            "experiment": {"name": "bad", "seed": 0},
            "task": {"name": "flow"},
            "model": {},
            "data": {},
            "training": {"max_steps": 1},
            "inference": {"num_steps": 1},
            "evaluation": {},
        }
        with self.assertRaises(ConfigError):
            validate_config(config)


class ExperimentDirectoryTest(unittest.TestCase):
    def test_standard_evidence_tree_is_created_without_overwrite(self):
        config = load_config(ROOT / "configs" / "smoke.yaml", expand_environment=False)
        with tempfile.TemporaryDirectory() as temporary:
            paths = create_experiment_directory(
                output_root=temporary,
                task="segmentation",
                experiment_name="unit-test",
                config=config,
                command=["python", "scripts/train.py", "--dry-run"],
                repo_root=ROOT,
            )
            for name in (
                "config.yaml",
                "command.txt",
                "git_commit.txt",
                "environment.txt",
                "train.log",
                "metrics.json",
            ):
                self.assertTrue((paths.root / name).is_file(), name)
            self.assertTrue(paths.checkpoints.is_dir())
            self.assertTrue(paths.predictions.is_dir())
            self.assertTrue(paths.visualizations.is_dir())
            with self.assertRaises(FileExistsError):
                create_experiment_directory(
                    output_root=temporary,
                    task="segmentation",
                    experiment_name="unit-test",
                    config=config,
                    command=["python"],
                    repo_root=ROOT,
                )


if __name__ == "__main__":
    unittest.main()
