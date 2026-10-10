#!/usr/bin/env python3
"""Focused offline checks for the pinned, experimental Maya recipe."""
from __future__ import annotations

import copy
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECIPE_ID = "a100-80g-hm/glm53-flash-maya-l"
sys.path.insert(0, str(ROOT / "scripts"))
import recipe


class MayaRecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = recipe.load(str(ROOT / "recipes.json"))
        cls.gpu, cls.models, cls.recipes = recipe._index(cls.registry)

    def test_trial_evidence_and_pins_are_separate_from_launch_status(self):
        r = self.recipes[RECIPE_ID]
        model = self.models[r["model"]]
        self.assertEqual(r["status"], "unmeasured")
        self.assertEqual(r["eta_min"], 110)
        self.assertEqual(self.gpu[r["gpu"]]["cu_per_hour"], 6.77)
        self.assertEqual(model["revision"], "f16ee4ba50d8fd2bb726ea1862ca6ca14ba188da")
        self.assertEqual(model["maya"]["commit"], "c189f7ac8114ca039e9643314d1fead014fc6ff5")
        evidence = recipe.facts(self.registry, RECIPE_ID)
        self.assertEqual(evidence["prefill_tps"]["v"], 1084.1)
        self.assertEqual(evidence["decode_tps"]["v"], 16.2)
        self.assertEqual(evidence["context"]["v"], 32768)
        self.assertFalse(evidence["vision"]["v"])

    def test_launch_contract_is_selectable_but_unverified(self):
        env = recipe.launch_env(self.registry, RECIPE_ID)
        self.assertEqual(env["ENGINE"], "maya")
        self.assertEqual(env["SHAPE"], "hm")
        self.assertEqual(env["MIN_VRAM_GIB"], "70")
        self.assertEqual(env["MAYA_REPO"], "https://github.com/mw00/project-maya.git")
        self.assertEqual(env["MAYA_MODEL"], "Maya-L")
        self.assertEqual(env["MAYA_MODEL_REVISION"], "f16ee4ba50d8fd2bb726ea1862ca6ca14ba188da")
        self.assertEqual(env["MAYA_MIN_DISK_GIB"], "180")
        self.assertEqual(env["MAYA_VISION"], "0")
        self.assertNotIn("MAYA_PORT", env)  # the engine inherits common PORT
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "recipe.py"), "env", RECIPE_ID],
            text=True, capture_output=True, env={k: v for k, v in os.environ.items() if k != "FORCE_RECIPE"},
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("ENGINE=maya", proc.stdout)

    def test_registry_requires_pinned_maya_and_hf_revisions(self):
        bad = copy.deepcopy(self.registry)
        bad_model = next(m for m in bad["models"] if m["id"] == "glm53-flash-maya-l")
        bad_model["maya"]["commit"] = "main"
        self.assertTrue(any("maya.commit" in issue for issue in recipe.validate(bad)))
        bad_model["maya"]["commit"] = "c189f7ac8114ca039e9643314d1fead014fc6ff5"
        bad_model["revision"] = "main"
        self.assertTrue(any("pinned 40-character commit" in issue for issue in recipe.validate(bad)))

    def test_experimental_recipe_lists_as_runnable_not_verified(self):
        flat = next(row for row in recipe.flat(self.registry) if row["id"] == RECIPE_ID)
        self.assertFalse(flat["verified"])
        self.assertTrue(flat["runnable"])


if __name__ == "__main__":
    unittest.main()
