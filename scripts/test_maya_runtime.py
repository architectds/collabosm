"""Offline safety checks for the Colab-only Maya adapter; never allocate a GPU."""
from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASH = shutil.which("bash")
if BASH and Path(BASH).parent.name.lower() == "windowsapps":
    BASH = None  # Windows' WSL launcher is not a Bash executable in this sandbox.


@unittest.skipUnless(BASH, "a usable Bash executable is required for VM script checks")
class MayaRuntimeTests(unittest.TestCase):
    def bash(self, script: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([BASH, "-c", script], cwd=ROOT, text=True,
                              capture_output=True, timeout=10)

    def test_vm_scripts_parse(self) -> None:
        for name in ("maya.sh", "bootstrap.sh", "serve.sh"):
            with self.subTest(script=name):
                result = self.bash(f"bash -n {name}")
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_disk_gate_fails_before_gpu_probe(self) -> None:
        result = self.bash(r'''
set -e
source ./maya.sh
say() { :; }
uname() { echo Linux; }
df() { printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/fake 100 1 100 1%% /content\n'; }
nvidia-smi() { echo CALLED >&2; return 0; }
MAYA_DATA_DIR=/content MAYA_MIN_DISK_GIB=180
if maya_preflight; then exit 2; fi
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("CALLED", result.stderr)

    def test_vram_gate_fails_before_any_install_or_start(self) -> None:
        result = self.bash(r'''
set -e
source ./maya.sh
say() { :; }
uname() { echo Linux; }
df() { printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/fake 200000000 1 200000000 1%% /content\n'; }
nvidia-smi() { echo 40960; }
MAYA_DATA_DIR=/content MAYA_MIN_DISK_GIB=100 MAYA_MIN_VRAM_GIB=70
if maya_preflight; then exit 2; fi
''')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_wrong_gpu_name_is_rejected_even_with_80g(self) -> None:
        result = self.bash(r'''
set -e
source ./maya.sh
say() { :; }
uname() { echo Linux; }
df() { printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/fake 200000000 1 200000000 1%% /content\n'; }
nvidia-smi() { if [[ "$*" == *memory.total* ]]; then echo 81920; else echo 'NVIDIA H100 80GB'; fi; }
MAYA_DATA_DIR=/content MAYA_MIN_DISK_GIB=100 MAYA_MIN_VRAM_GIB=70
if maya_preflight; then exit 2; fi
''')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_partial_download_can_resume_without_original_free_space(self) -> None:
        result = self.bash(r'''
set -e
source ./maya.sh
say() { :; }
uname() { echo Linux; }
df() { printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/fake 200000000 1 40000000 1%% /content\n'; }
nvidia-smi() { if [[ "$*" == *memory.total* ]]; then echo 81920; else echo 'NVIDIA A100-SXM4-80GB'; fi; }
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/Maya-L"
touch "$tmp/Maya-L/GLM-5.3-Flash-Maya-L-IQ3_S-00001-of-00004.gguf"
MAYA_MODEL_DIR="$tmp" MAYA_DATA_DIR=/content MAYA_MIN_DISK_GIB=180 MAYA_MIN_VRAM_GIB=70
maya_preflight
''')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_secret_not_passed_as_cli_argument_or_logged(self) -> None:
        text = (ROOT / "maya.sh").read_text(encoding="utf-8")
        self.assertIn('STRATA_API_KEY="$KEY"', text)
        self.assertIn('maya.py --config "$MAYA_CONFIG" --port "$PORT" --plain', text)
        self.assertNotIn("--api-key", text)
        self.assertNotIn('say "$KEY"', text)

    def test_source_and_model_are_pinned_and_revision_patch_is_repeatable(self) -> None:
        text = (ROOT / "maya.sh").read_text(encoding="utf-8")
        self.assertIn("c189f7ac8114ca039e9643314d1fead014fc6ff5", text)
        self.assertIn("f16ee4ba50d8fd2bb726ea1862ca6ca14ba188da", text)
        self.assertIn('m.group(2) not in ("main", revision)', text)
        self.assertIn("if m.group(2) == \"main\":", text)


if __name__ == "__main__":
    unittest.main()
