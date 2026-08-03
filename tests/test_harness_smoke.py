from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class HarnessSmokeTest(unittest.TestCase):
    def test_core_harness_files_exist(self):
        required = [
            "AGENTS.md",
            "docs/agent-harness/index.md",
            "docs/agent-harness/quality.md",
            "docs/agent-harness/tools.md",
            "docs/agent-harness/review.md",
            "agents/skills/README.md",
        ]
        missing = [path for path in required if not (ROOT / path).exists()]
        self.assertEqual([], missing)


if __name__ == "__main__":
    unittest.main()
