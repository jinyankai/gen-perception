import unittest
from types import SimpleNamespace

import torch
from torch import nn

from scripts.operator.segmentation_ddp_training_gate import (
    validate_replica_measurements,
)
from scripts.operator.segmentation_training_gate import (
    _optimizer_groups,
    _segmentation_sections,
)


class _Conditioner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.task_embeddings = nn.Embedding(2, 4)
        self.adapters = nn.ModuleDict({"segmentation": nn.Linear(4, 4)})


class OperatorTrainingGateTest(unittest.TestCase):
    def test_optimizer_groups_are_disjoint_and_use_configured_rates(self):
        denoiser = SimpleNamespace(
            shared_unet=nn.Conv2d(8, 4, kernel_size=1),
            conditioner=_Conditioner(),
        )
        training = {
            "optimizer": {
                "shared_unet_lr": 1.0e-5,
                "conditioner_lr": 2.0e-5,
                "adapter_lr": 3.0e-5,
            }
        }
        groups, parameters = _optimizer_groups(denoiser, training)

        self.assertEqual(
            ["shared_unet", "conditioner", "condition_adapter"],
            [group["group_name"] for group in groups],
        )
        self.assertEqual([1.0e-5, 2.0e-5, 3.0e-5], [group["lr"] for group in groups])
        self.assertEqual(len(parameters), len({id(parameter) for parameter in parameters}))
        self.assertTrue(all(isinstance(parameter, torch.nn.Parameter) for parameter in parameters))

    def test_segmentation_sections_accept_single_and_multitask_configs(self):
        single = {
            "task": {"name": "segmentation"},
            "data": {"dataset": "ade20k"},
            "evaluation": {"protocol": "ade20k_semantic_150"},
        }
        multitask = {
            "task": {"name": "multitask"},
            "data": {"datasets": {"segmentation": {"dataset": "ade20k"}}},
            "evaluation": {
                "segmentation": {"protocol": "ade20k_semantic_150"}
            },
        }

        self.assertEqual((single["data"], single["evaluation"]), _segmentation_sections(single))
        self.assertEqual(
            (
                multitask["data"]["datasets"]["segmentation"],
                multitask["evaluation"]["segmentation"],
            ),
            _segmentation_sections(multitask),
        )

    def test_ddp_replica_measurements_require_updates_and_synced_parameters(self):
        summary = validate_replica_measurements(
            [0.1, 0.1],
            [4.0, 4.0],
            [8.0, 8.0],
        )
        self.assertEqual(0.0, summary["checksum_spread"])
        with self.assertRaises(FloatingPointError):
            validate_replica_measurements(
                [0.1, 0.2],
                [4.0, 5.0],
                [8.0, 9.0],
            )


if __name__ == "__main__":
    unittest.main()
