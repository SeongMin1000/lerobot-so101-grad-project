#!/usr/bin/env python3
"""
test_same_color_pcgrad.py

Unit test and sanity check suite for Same-Color Paired PCGrad Expert-Only Fine-Tuning:
  A. dot > 0: No projection, g1_pc == g1, g2_pc == g2.
  B. dot < 0: Canonical PCGrad projection matches exact math and orthogonality (g1_pc . g2 == 0, g2_pc . g1 == 0).
  C. Zero norm / eps: Handles without NaN.
  D. None gradients: Handled gracefully.
  E. PCGrad off: Final gradient == 0.5 * (g1 + g2).
  F. PCGrad on with dot >= 0: Identical to PCGrad off.
  G. Same-color sampler: Guarantees Task 1 and Task 2 never have different colors.
  H. 5-step color cycle: Exactly 20% balanced over multiples of 5 steps.
  I. Batch size: Exactly 16 samples per task per step.
  J. Exact 822 normalization stats loading fidelity.
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

LEROBOT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LEROBOT_ROOT))
sys.path.insert(0, str(LEROBOT_ROOT / "project/scripts/train"))

from train_smolvla_same_color_pcgrad import (
    COLOR_NAMES,
    TASK1_EPISODE_RANGES,
    TASK2_EPISODE_RANGES,
    SameColorBalancedSampler,
    compute_pcgrad,
    load_model_exact_stats,
    run_dataset_sampler_dryrun,
    save_compatible_checkpoint,
    verify_dataset_and_configuration,
)


class TestCanonicalPCGradMath(unittest.TestCase):
    def test_a_positive_dot_no_projection(self):
        """A. dot > 0: No projection."""
        g1 = [torch.tensor([1.0, 2.0, 3.0])]
        g2 = [torch.tensor([2.0, 4.0, 6.0])]
        g1_pc, g2_pc, dot, cos, conflict = compute_pcgrad(g1, g2)

        self.assertFalse(conflict)
        self.assertGreater(dot, 0.0)
        self.assertAlmostEqual(cos, 1.0, places=5)
        self.assertTrue(torch.allclose(g1_pc[0], g1[0]))
        self.assertTrue(torch.allclose(g2_pc[0], g2[0]))

    def test_b_negative_dot_orthogonal_projection(self):
        """B. dot < 0: Canonical PCGrad projection matches exact math."""
        # g1 = [1.0, 0.0], g2 = [-1.0, 1.0]
        # dot = -1.0
        # ||g2||^2 = 2.0, scale1 = -1.0 / 2.0 = -0.5
        # g1_pc = [1.0, 0.0] - (-0.5) * [-1.0, 1.0] = [1.0 - 0.5, 0.0 + 0.5] = [0.5, 0.5]
        # Check orthogonality: g1_pc . g2 = 0.5 * (-1.0) + 0.5 * 1.0 = 0.0
        # ||g1||^2 = 1.0, scale2 = -1.0 / 1.0 = -1.0
        # g2_pc = [-1.0, 1.0] - (-1.0) * [1.0, 0.0] = [0.0, 1.0]
        # Check orthogonality: g2_pc . g1 = 0.0 * 1.0 + 1.0 * 0.0 = 0.0
        g1 = [torch.tensor([1.0, 0.0])]
        g2 = [torch.tensor([-1.0, 1.0])]

        g1_pc, g2_pc, dot, cos, conflict = compute_pcgrad(g1, g2)

        self.assertTrue(conflict)
        self.assertAlmostEqual(dot, -1.0, places=5)

        expected_g1_pc = torch.tensor([0.5, 0.5])
        expected_g2_pc = torch.tensor([0.0, 1.0])

        self.assertTrue(torch.allclose(g1_pc[0], expected_g1_pc, atol=1e-5))
        self.assertTrue(torch.allclose(g2_pc[0], expected_g2_pc, atol=1e-5))

        # Check exact orthogonality
        dot_g1pc_g2 = torch.dot(g1_pc[0], g2[0]).item()
        dot_g2pc_g1 = torch.dot(g2_pc[0], g1[0]).item()
        self.assertAlmostEqual(dot_g1pc_g2, 0.0, places=5)
        self.assertAlmostEqual(dot_g2pc_g1, 0.0, places=5)

    def test_c_zero_norm_handling(self):
        """C. Zero norm: Handles gracefully without NaN."""
        g1 = [torch.tensor([0.0, 0.0, 0.0])]
        g2 = [torch.tensor([1.0, 2.0, 3.0])]

        g1_pc, g2_pc, dot, cos, conflict = compute_pcgrad(g1, g2)

        self.assertFalse(conflict)
        self.assertFalse(math.isnan(dot))
        self.assertFalse(math.isnan(cos))
        self.assertFalse(torch.isnan(g1_pc[0]).any())
        self.assertFalse(torch.isnan(g2_pc[0]).any())

    def test_d_none_gradient_handling(self):
        """D. None gradient: Handled without crashing."""
        g1 = [torch.tensor([1.0, 2.0]), None]
        g2 = [torch.tensor([2.0, 3.0]), None]

        g1_pc, g2_pc, dot, cos, conflict = compute_pcgrad(g1, g2)
        self.assertIsNone(g1_pc[1])
        self.assertIsNone(g2_pc[1])
        self.assertFalse(conflict)

    def test_e_and_f_pcgrad_off_vs_pcgrad_on_non_conflicting(self):
        """E & F. Non-conflicting PCGrad matches standard 0.5*(g1+g2)."""
        g1 = [torch.tensor([2.0, 4.0, 6.0])]
        g2 = [torch.tensor([1.0, 3.0, 5.0])]

        g1_pc, g2_pc, dot, cos, conflict = compute_pcgrad(g1, g2)
        self.assertFalse(conflict)

        # Baseline equivalence: 0.5 * (g1 + g2)
        g_final_pcgrad = [0.5 * (p1 + p2) for p1, p2 in zip(g1_pc, g2_pc)]
        g_final_standard = [0.5 * (p1 + p2) for p1, p2 in zip(g1, g2)]

        self.assertTrue(torch.allclose(g_final_pcgrad[0], g_final_standard[0]))


class TestSameColorBalancedSampler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p1 = LEROBOT_ROOT / "outputs/gradient_conflict_comparison_865_vs_822"
        # Use existing dataset parquet files from local HF cache
        t1_path = Path.home() / ".cache/huggingface/lerobot/eslab1234/smolvla_task1_hil_575_285k_v4_120ep_trimmed_merged"
        t2_path = Path.home() / ".cache/huggingface/lerobot/eslab1234/smolvla_task2_hil_575_285k_v4_127ep_trimmed_merged"

        cls.ep_p1 = t1_path / "meta/episodes/chunk-000/file-000.parquet"
        cls.ep_p2 = t2_path / "meta/episodes/chunk-000/file-000.parquet"

        if cls.ep_p1.exists() and cls.ep_p2.exists():
            cls.sampler = SameColorBalancedSampler(
                t1_parquet=cls.ep_p1,
                t2_parquet=cls.ep_p2,
                t1_ranges=TASK1_EPISODE_RANGES,
                t2_ranges=TASK2_EPISODE_RANGES,
                batch_size_per_task=16,
                seed=42,
            )
        else:
            cls.sampler = None

    def test_g_and_i_same_color_pairing_and_batch_size(self):
        """G & I. Same-color pairing and batch size = 16."""
        if self.sampler is None:
            self.skipTest("Local dataset parquet files not found.")

        for _ in range(25):
            color, idx_t1, idx_t2 = self.sampler.sample_step()
            self.assertIn(color, COLOR_NAMES)
            self.assertEqual(len(idx_t1), 16)
            self.assertEqual(len(idx_t2), 16)

            # Verify that idx_t1 frames belong strictly to the selected color's episodes
            # and idx_t2 frames belong strictly to the selected color's episodes
            t1_start_ep, t1_end_ep = TASK1_EPISODE_RANGES[color]
            t2_start_ep, t2_end_ep = TASK2_EPISODE_RANGES[color]

            # All frames in idx_t1 must be within valid frame pools of color
            t1_valid_frames = set()
            for ep in range(t1_start_ep, t1_end_ep + 1):
                t1_valid_frames.update(self.sampler.t1_pools[color][ep])

            t2_valid_frames = set()
            for ep in range(t2_start_ep, t2_end_ep + 1):
                t2_valid_frames.update(self.sampler.t2_pools[color][ep])

            for f in idx_t1:
                self.assertIn(f, t1_valid_frames)
            for f in idx_t2:
                self.assertIn(f, t2_valid_frames)

    def test_h_5_step_cycle_balance(self):
        """H. Exactly 20% balance across any multiple of 5 steps."""
        if self.sampler is None:
            self.skipTest("Local dataset parquet files not found.")

        color_counts = {c: 0 for c in COLOR_NAMES}
        num_cycles = 20  # 100 steps
        for _ in range(num_cycles * 5):
            color, _, _ = self.sampler.sample_step()
            color_counts[color] += 1

        for c, count in color_counts.items():
            self.assertEqual(count, num_cycles, f"Color {c} not sampled exactly {num_cycles} times!")


class TestNormalizationFidelity(unittest.TestCase):
    def test_j_822_normalizer_stats_loaded(self):
        """J. 822 training-time statistics fidelity."""
        policy_path = "eslab1234/smolvla_multitask_5blocks_v3_822ep_from865_expertonly_b32_lr5e6_50k"
        import logging
        logger = logging.getLogger("test")

        try:
            stats = load_model_exact_stats(policy_path, logger)
        except Exception as e:
            self.skipTest(f"Hub or cache not accessible: {e}")

        self.assertIn("action", stats)
        self.assertIn("observation.state", stats)
        self.assertIn("mean", stats["action"])
        self.assertIn("std", stats["action"])

        action_mean = stats["action"]["mean"].tolist()
        # Verify against exact known 822 serialized mean: [-3.411, -16.869, 2.100, ...]
        self.assertAlmostEqual(action_mean[0], -3.4109368, places=3)
        self.assertAlmostEqual(action_mean[1], -16.868828, places=3)
        self.assertAlmostEqual(action_mean[2], 2.1000278, places=3)


class TestDatasetVerificationAndSamplerDryRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        cls.t1_path = Path.home() / ".cache/huggingface/lerobot/eslab1234/smolvla_task1_hil_575_285k_v4_120ep_trimmed_merged"
        cls.t2_path = Path.home() / ".cache/huggingface/lerobot/eslab1234/smolvla_task2_hil_575_285k_v4_127ep_trimmed_merged"

        if cls.t1_path.exists() and cls.t2_path.exists():
            cls.ds_t1 = LeRobotDataset("eslab1234/smolvla_task1_hil_575_285k_v4_120ep_trimmed_merged", root=cls.t1_path)
            cls.ds_t2 = LeRobotDataset("eslab1234/smolvla_task2_hil_575_285k_v4_127ep_trimmed_merged", root=cls.t2_path)
            ep_p1 = cls.t1_path / "meta/episodes/chunk-000/file-000.parquet"
            ep_p2 = cls.t2_path / "meta/episodes/chunk-000/file-000.parquet"
            cls.sampler = SameColorBalancedSampler(
                t1_parquet=ep_p1,
                t2_parquet=ep_p2,
                t1_ranges=TASK1_EPISODE_RANGES,
                t2_ranges=TASK2_EPISODE_RANGES,
                batch_size_per_task=16,
                seed=42,
            )
        else:
            cls.ds_t1 = None
            cls.ds_t2 = None
            cls.sampler = None

    def test_k_dataset_verification_change14(self):
        """K. CHANGE 14: Verifies episode counts and catches forbidden datasets."""
        if self.ds_t1 is None or self.ds_t2 is None:
            self.skipTest("Local datasets not found.")

        import logging
        logger = logging.getLogger("test_verify")

        # 1. Valid configuration must pass cleanly
        verify_dataset_and_configuration(
            task1_dataset="eslab1234/smolvla_task1_hil_575_285k_v4_120ep_trimmed_merged",
            task2_dataset="eslab1234/smolvla_task2_hil_575_285k_v4_127ep_trimmed_merged",
            ds_t1=self.ds_t1,
            ds_t2=self.ds_t2,
            t1_ranges=TASK1_EPISODE_RANGES,
            t2_ranges=TASK2_EPISODE_RANGES,
            logger=logger,
        )

        # 2. Forbidden 575 dataset must raise ValueError
        with self.assertRaises(ValueError):
            verify_dataset_and_configuration(
                task1_dataset="eslab1234/multitask_5blocks_v3_575ep_merged",
                task2_dataset="eslab1234/smolvla_task2_hil_575_285k_v4_127ep_trimmed_merged",
                ds_t1=self.ds_t1,
                ds_t2=self.ds_t2,
                t1_ranges=TASK1_EPISODE_RANGES,
                t2_ranges=TASK2_EPISODE_RANGES,
                logger=logger,
            )

    def test_l_dataset_sampler_dryrun_change15(self):
        """L. CHANGE 15: Verifies 10-step sampler dry-run with prompt and color checking."""
        if self.sampler is None:
            self.skipTest("Local datasets not found.")

        import logging
        logger = logging.getLogger("test_dryrun")
        run_dataset_sampler_dryrun(
            sampler=self.sampler,
            ds_t1=self.ds_t1,
            ds_t2=self.ds_t2,
            t1_ranges=TASK1_EPISODE_RANGES,
            t2_ranges=TASK2_EPISODE_RANGES,
            dryrun_steps=10,
            logger=logger,
        )

    def test_m_checkpoint_layout_change18(self):
        """M. CHANGE 18: Verifies standard checkpoint directory structure and 'last' symlink."""
        import tempfile
        import logging
        from unittest.mock import MagicMock

        logger = logging.getLogger("test_ckpt")
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            total_steps = 15000
            step = 2500

            # Mock policy, optimizer, and scheduler
            mock_policy = MagicMock()
            mock_policy.save_pretrained = MagicMock(side_effect=lambda p: (p / "model.safetensors").touch())

            param = torch.nn.Parameter(torch.randn(2, 2))
            opt = torch.optim.AdamW([param], lr=1e-4)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: 1.0)

            # Dummy base directory with required artifacts
            base_dir = out_dir / "dummy_base"
            base_dir.mkdir(parents=True)
            for fname in [
                "policy_preprocessor.json",
                "policy_preprocessor_step_5_normalizer_processor.safetensors",
                "policy_postprocessor.json",
                "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
                "train_config.json",
            ]:
                (base_dir / fname).write_text("{}")

            saved_dir = save_compatible_checkpoint(
                out_dir=out_dir,
                total_steps=total_steps,
                step=step,
                policy=mock_policy,
                optimizer=opt,
                scheduler=sched,
                base_policy_path=str(base_dir),
                logger=logger,
            )

            # 1. Check directory path matches get_step_checkpoint_dir
            expected_dir = out_dir / "checkpoints" / "002500"
            self.assertEqual(saved_dir, expected_dir)
            self.assertTrue(expected_dir.exists())

            # 2. Check pretrained_model directory contents
            pretrained_dir = expected_dir / "pretrained_model"
            self.assertTrue(pretrained_dir.exists())
            self.assertTrue((pretrained_dir / "model.safetensors").exists())
            self.assertTrue((pretrained_dir / "policy_preprocessor.json").exists())
            self.assertTrue((pretrained_dir / "policy_preprocessor_step_5_normalizer_processor.safetensors").exists())
            self.assertTrue((pretrained_dir / "policy_postprocessor.json").exists())
            self.assertTrue((pretrained_dir / "train_config.json").exists())

            # 3. Check training_state directory contents
            state_dir = expected_dir / "training_state"
            self.assertTrue(state_dir.exists())
            self.assertTrue((state_dir / "training_step.json").exists())
            self.assertTrue((state_dir / "optimizer_state.safetensors").exists())
            self.assertTrue((state_dir / "scheduler_state.json").exists())
            self.assertTrue((state_dir / "rng_state.safetensors").exists())

            # 4. Check checkpoints/last symlink
            last_link = out_dir / "checkpoints" / "last"
            self.assertTrue(last_link.is_symlink())
            self.assertEqual(last_link.resolve(), expected_dir.resolve())


if __name__ == "__main__":
    unittest.main()
