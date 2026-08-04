import tempfile
import unittest
from pathlib import Path

from perception_diffusion.utils.config import (
    ConfigError,
    configured_task_names,
    load_config,
    validate_config,
)
from perception_diffusion.utils.experiment import create_experiment_directory


ROOT = Path(__file__).resolve().parents[1]


class ConfigTest(unittest.TestCase):
    def test_smoke_config_loads(self):
        config = load_config(ROOT / "configs" / "smoke.yaml", expand_environment=False)
        self.assertEqual("segmentation", config["task"]["name"])
        self.assertEqual(
            "stable-diffusion-2", config["model"]["backbone"]["family"]
        )
        self.assertEqual(
            "task_token_cross_attention", config["model"]["conditioning"]["type"]
        )

    def test_multitask_config_selects_three_canonical_tasks(self):
        config = load_config(
            ROOT / "configs" / "multitask" / "stage1_shared_unet.yaml",
            expand_environment=False,
        )
        self.assertEqual(
            ("segmentation", "depth", "normal"), configured_task_names(config)
        )

    def test_multitask_config_rejects_duplicate_tasks(self):
        config = load_config(
            ROOT / "configs" / "multitask" / "stage1_shared_unet.yaml",
            expand_environment=False,
        )
        config["task"]["tasks"] = ["depth", "depth"]
        with self.assertRaises(ConfigError):
            validate_config(config)

    def test_learnable_target_adapter_ablation_is_explicit(self):
        config = load_config(
            ROOT / "configs" / "ablations" / "learnable_pre_vae_adapter.yaml",
            expand_environment=False,
        )
        self.assertTrue(config["model"]["target_adapter"]["enabled"])
        self.assertEqual("pre_vae", config["model"]["target_adapter"]["placement"])

    def test_gt_present_segmentation_query_mode_is_rejected(self):
        config = load_config(ROOT / "configs" / "smoke.yaml", expand_environment=False)
        config["evaluation"]["query"]["mode"] = "gt_present_classes"
        with self.assertRaises(ConfigError):
            validate_config(config)

    def test_ade20k_evaluation_target_query_cannot_inspect_gt(self):
        config = load_config(
            ROOT / "configs" / "segmentation" / "ade20k.yaml",
            expand_environment=False,
        )
        config["data"]["evaluation_query_sampling"] = "first_present"
        with self.assertRaises(ConfigError):
            validate_config(config)

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

    def test_supported_prediction_types_pass_validation(self):
        for prediction_type in ("epsilon", "v_prediction", "sample"):
            config = load_config(
                ROOT / "configs" / "smoke.yaml", expand_environment=False
            )
            config["model"]["prediction_type"] = prediction_type
            validate_config(config)

    def test_invalid_prediction_type_is_rejected(self):
        config = load_config(ROOT / "configs" / "smoke.yaml", expand_environment=False)
        config["model"]["prediction_type"] = "not-a-real-objective"
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
