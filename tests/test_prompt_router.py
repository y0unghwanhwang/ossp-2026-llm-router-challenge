# SPDX-FileCopyrightText: Copyright 2026 황영환
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest import mock

from ossp_router import prompt_router
from ossp_router.heuristic import episode_text
from ossp_router.protocol import (
    MODEL_IDS,
    ProtocolError,
    load_bundled_policy,
    load_input,
    load_submission,
    parse_input,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _changed_batch(original):
    return parse_input(
        {
            "schema_version": original.schema_version,
            "challenge_id": original.challenge_id,
            "split": original.split,
            "episodes": [
                {
                    "episode_id": f"changed-{index}",
                    **(
                        {"prompt": episode.prompt}
                        if episode.prompt is not None
                        else {
                            "messages": [
                                {"role": item.role, "content": item.content}
                                for item in episode.messages or ()
                            ]
                        }
                    ),
                }
                for index, episode in enumerate(reversed(original.episodes))
            ],
        }
    )


def _by_content(inputs, submission):
    models = {item.episode_id: item.model_id for item in submission.decisions}
    return {
        episode_text(episode): models[episode.episode_id]
        for episode in inputs.episodes
    }


def _prediction(fingerprint: int, *, ood: bool = False):
    return prompt_router.EpisodePrediction(
        text_fingerprint=fingerprint,
        primary_delta_ax31=0.2,
        primary_delta_think=0.1,
        companion_delta_ax31=0.1,
        companion_delta_think=0.05,
        primary_prob_ax31=0.8,
        primary_prob_think=0.8,
        companion_prob_ax31=0.7,
        companion_prob_think=0.7,
        mean_costs={
            prompt_router.LIGHT: 1.0,
            prompt_router.AX31: 2.0,
            prompt_router.THINK: 5.0,
        },
        is_ood=ood,
    )


class PromptRouterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_bundled_policy()
        cls.artifact = prompt_router.load_artifact()
        cls.toy = load_input(ROOT / "data/toy/inputs.json")

    def test_bundled_artifact_matches_policy_and_contains_no_rows(self) -> None:
        self.assertEqual(self.policy.policy_id, self.artifact.policy_id)
        raw = (ROOT / "src/ossp_router/resources/prompt-router-public.v1.json").read_text(
            encoding="utf-8"
        )
        for episode in self.toy.episodes:
            self.assertNotIn(episode.episode_id, raw)
            self.assertNotIn(episode_text(episode), raw)

    def test_artifact_parser_rejects_unknown_fields(self) -> None:
        path = ROOT / "src/ossp_router/resources/prompt-router-public.v1.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["prompt_lookup"] = {}
        with self.assertRaises(ProtocolError):
            prompt_router.parse_artifact(raw)

    def test_prompt_and_messages_features_are_finite(self) -> None:
        batch = parse_input(
            {
                "schema_version": 1,
                "challenge_id": "feature-test",
                "split": "test",
                "episodes": [
                    {"episode_id": "p", "prompt": "Prove that 2 + 2 = 4."},
                    {
                        "episode_id": "m",
                        "messages": [
                            {"role": "system", "content": "Answer briefly."},
                            {"role": "user", "content": "왜 2+2는 4인가?"},
                        ],
                    },
                ],
            }
        )
        vectors = [prompt_router.raw_feature_vector(item, 16, 16) for item in batch.episodes]
        self.assertEqual(2, len(vectors))
        self.assertEqual(len(prompt_router.DENSE_FEATURE_NAMES) + 32, len(vectors[0]))
        self.assertNotEqual(vectors[0], vectors[1])

    def test_ids_and_order_do_not_change_content_decisions(self) -> None:
        changed = _changed_batch(self.toy)
        for tier in ("fast", "balanced", "premium"):
            with self.subTest(tier=tier):
                original = prompt_router.make_prompt_submission(
                    self.toy, self.policy, self.artifact, tier
                ).submission
                reordered = prompt_router.make_prompt_submission(
                    changed, self.policy, self.artifact, tier
                ).submission
                self.assertEqual(
                    _by_content(self.toy, original),
                    _by_content(changed, reordered),
                )

    def test_all_episodes_appear_once_with_allowed_models(self) -> None:
        for tier in ("fast", "balanced", "premium"):
            submission = prompt_router.make_prompt_submission(
                self.toy, self.policy, self.artifact, tier
            ).submission
            ids = [item.episode_id for item in submission.decisions]
            self.assertEqual(len(self.toy.episodes), len(ids))
            self.assertEqual(len(ids), len(set(ids)))
            self.assertTrue(all(item.model_id in MODEL_IDS for item in submission.decisions))

    def test_allocator_keeps_conservative_batch_limit_and_breaks_ties_by_content(self) -> None:
        inputs = parse_input(
            {
                "schema_version": 1,
                "challenge_id": "allocator-test",
                "split": "test",
                "episodes": [
                    {"episode_id": "z", "prompt": "gamma"},
                    {"episode_id": "x", "prompt": "alpha"},
                    {"episode_id": "y", "prompt": "beta"},
                ],
            }
        )
        length = len(prompt_router.DENSE_FEATURE_NAMES) + 32
        zero = prompt_router.LinearHead(0.0, (0.0,) * length)
        artifact = prompt_router.PromptRouterArtifact(
            word_bins=16,
            char_bins=16,
            feature_mean=(0.0,) * length,
            feature_scale=(1.0,) * length,
            primary_heads={name: zero for name in prompt_router.PRIMARY_HEAD_NAMES},
            companion_heads={name: zero for name in prompt_router.COMPANION_HEAD_NAMES},
            cost_bounds=prompt_router.CostBounds(
                1.0, dict.fromkeys(MODEL_IDS, 1.0)
            ),
            ood_thresholds=prompt_router.OodThresholds(10.0, 10.0),
            tier_policies={
                "fast": prompt_router.TierPolicy(1.20, 0.5),
                "balanced": prompt_router.TierPolicy(1.60, 0.5),
                "premium": prompt_router.TierPolicy(3.0, 0.5),
            },
            policy_id=self.policy.policy_id,
            policy_digest=prompt_router.policy_sha256(self.policy),
            training_summary={},
        )
        predictions = tuple(
            _prediction(prompt_router._stable_hash(episode_text(item)))
            for item in inputs.episodes
        )
        selected, ratio, _fallback, _ood = prompt_router.allocate_predictions(
            inputs, self.policy, artifact, "balanced", predictions
        )
        self.assertLessEqual(ratio, 1.60)
        self.assertLessEqual(
            prompt_router.conservative_budget_slack(
                selected,
                predictions,
                artifact,
                float(self.policy.tiers["balanced"].budget_multiplier),
            ),
            0.0,
        )
        self.assertEqual(1, selected.count(prompt_router.AX31))

        changed = _changed_batch(inputs)
        changed_predictions = tuple(
            _prediction(prompt_router._stable_hash(episode_text(item)))
            for item in changed.episodes
        )
        reordered, _, _, _ = prompt_router.allocate_predictions(
            changed, self.policy, artifact, "balanced", changed_predictions
        )
        self.assertEqual(
            {episode_text(item): model for item, model in zip(inputs.episodes, selected)},
            {episode_text(item): model for item, model in zip(changed.episodes, reordered)},
        )

    def test_ood_episode_uses_tierfixed_fallback(self) -> None:
        predictions = tuple(_prediction(index, ood=True) for index, _ in enumerate(self.toy.episodes))
        for tier in ("fast", "balanced", "premium"):
            selected, _ratio, fallback, ood = prompt_router.allocate_predictions(
                self.toy, self.policy, self.artifact, tier, predictions
            )
            self.assertEqual(
                (prompt_router.TIER_MODEL[tier],) * len(self.toy.episodes), selected
            )
            self.assertEqual(len(self.toy.episodes), fallback)
            self.assertEqual(len(self.toy.episodes), ood)

    def test_missing_artifact_cli_falls_back_to_c2_without_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "premium.json"
            code = prompt_router.main(
                [
                    "--input",
                    str(ROOT / "data/toy/inputs.json"),
                    "--tier",
                    "premium",
                    "--artifact",
                    str(pathlib.Path(directory) / "missing.json"),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(0, code)
            submission = load_submission(output)
            self.assertTrue(
                all(item.model_id == prompt_router.AX31 for item in submission.decisions)
            )

    def test_corrupt_or_wrong_schema_artifact_cli_is_exact_c2_for_every_tier(self) -> None:
        artifact_path = ROOT / "src/ossp_router/resources/prompt-router-public.v1.json"
        valid = json.loads(artifact_path.read_text(encoding="utf-8"))
        for mutation in ("corrupt-json", "wrong-schema"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = pathlib.Path(directory)
                artifact = root / "artifact.json"
                if mutation == "corrupt-json":
                    artifact.write_text("{", encoding="utf-8")
                else:
                    wrong = dict(valid)
                    wrong["schema_version"] = 999
                    artifact.write_text(json.dumps(wrong), encoding="utf-8")
                for tier in ("fast", "balanced", "premium"):
                    output = root / f"{mutation}-{tier}.json"
                    code = prompt_router.main(
                        [
                            "--input",
                            str(ROOT / "data/toy/inputs.json"),
                            "--tier",
                            tier,
                            "--artifact",
                            str(artifact),
                            "--output",
                            str(output),
                        ]
                    )
                    self.assertEqual(0, code)
                    self.assertEqual(
                        prompt_router.make_tierfixed_submission(
                            self.toy, self.policy, tier
                        ),
                        load_submission(output),
                    )

    def test_numeric_prediction_failure_is_exact_c2_for_every_tier(self) -> None:
        with mock.patch.object(
            prompt_router,
            "predict_episode",
            side_effect=OverflowError("synthetic abnormal cost estimate"),
        ):
            for tier in ("fast", "balanced", "premium"):
                plan = prompt_router.make_prompt_submission(
                    self.toy, self.policy, self.artifact, tier
                )
                self.assertEqual(
                    prompt_router.make_tierfixed_submission(
                        self.toy, self.policy, tier
                    ),
                    plan.submission,
                )
                self.assertEqual(len(self.toy.episodes), plan.fallback_count)

    @unittest.skipUnless(
        (ROOT / "data/materialized/dev/inputs.json").is_file(),
        "materialized Dev가 있을 때만 프롬프트 선택 비자명성을 검사",
    )
    def test_public_dev_premium_decisions_are_prompt_dependent(self) -> None:
        inputs = load_input(ROOT / "data/materialized/dev/inputs.json")
        submission = prompt_router.make_prompt_submission(
            inputs, self.policy, self.artifact, "premium"
        ).submission
        models = {item.model_id for item in submission.decisions}
        self.assertGreater(len(models), 1)
        self.assertIn(prompt_router.LIGHT, models)
        self.assertIn(prompt_router.THINK, models)
        changed = _changed_batch(inputs)
        reordered = prompt_router.make_prompt_submission(
            changed, self.policy, self.artifact, "premium"
        ).submission
        self.assertEqual(
            _by_content(inputs, submission),
            _by_content(changed, reordered),
        )


if __name__ == "__main__":
    unittest.main()
