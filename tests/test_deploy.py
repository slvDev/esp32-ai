"""scripts/deploy.sh: what it selects, what it refuses, and what order it does
things in.

Nothing here touches a board. Each test builds a throwaway repository holding
the script and stub tools, puts fakes for arduino-cli, esptool and uv on PATH,
and points PORT at a file rather than a device. Every stub appends to one log,
so the assertions are about the order and the arguments of real invocations
rather than about what the script prints.

The property that matters most is the last one: the firmware is compiled before
either image is written, so a build failure cannot leave new weights running
under old firmware.

  uv run python -m unittest discover -s tests
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "deploy.sh"

# Each stub records the call and succeeds. `uv` also has to produce the headers
# the real generators would, since the compile step is told to read them.
STUB = """#!/usr/bin/env bash
echo "{name} $*" >> "$CALL_LOG"
{body}
exit 0
"""

ARDUINO_BODY = """
if [ "$1" = "compile" ]; then echo "Sketch uses 1234 bytes"; fi
if [ "$1" = "upload" ]; then echo "New upload port"; fi
"""

# The gates are compiled and then run, so the compiler stub has to leave
# something executable where -o pointed.
CC_BODY = """
prev=""
for a in "$@"; do
  [ "$prev" = "-o" ] && { printf '#!/bin/sh\\necho PASS\\nexit 0\\n' > "$a"; chmod +x "$a"; }
  prev=$a
done
"""

UV_BODY = """
# Mimic the generators: emit the "wrote" line deploy.sh keeps, and touch the
# output so a later step cannot pass on a stale file.
out=""; prev=""
for a in "$@"; do
  case "$prev" in --out|--out-dir) out=$a ;; esac
  prev=$a
done
if [ -n "$out" ]; then mkdir -p "$(dirname "$out")" 2>/dev/null || true; : > "$out" 2>/dev/null || true; fi
echo "wrote ${out:-nothing}"
echo "PASS"
"""


class DeployHarness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name) / "repo"
        (self.repo / "scripts").mkdir(parents=True)
        (self.repo / "scripts" / "deploy.sh").write_text(SCRIPT.read_text())
        os.chmod(self.repo / "scripts" / "deploy.sh", 0o755)

        # Only the paths deploy.sh reads have to exist.
        for sketch in ("esp32_tinystories", "esp32_barista"):
            d = self.repo / "firmware" / sketch / "tools"
            d.mkdir(parents=True)
            (self.repo / "firmware" / sketch / "generated").mkdir()
        (self.repo / "runtime" / "host_verify").mkdir(parents=True)
        for f in ("verify.c", "staging_verify.c"):
            (self.repo / "runtime" / "host_verify" / f).write_text(
                "int main(void){return 0;}\n")

        self.bin = Path(self.tmp.name) / "bin"
        self.bin.mkdir()
        self.log = Path(self.tmp.name) / "calls.log"
        for name, body in (("arduino-cli", ARDUINO_BODY), ("esptool", ""),
                           ("uv", UV_BODY), ("cc", CC_BODY)):
            p = self.bin / name
            p.write_text(STUB.format(name=name, body=body))
            os.chmod(p, 0o755)

        self.port = Path(self.tmp.name) / "port"
        self.port.write_text("")

    def tearDown(self):
        self.tmp.cleanup()

    def artifacts(self, model, files):
        d = self.repo / "artifacts" / model
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            (d / f).write_text("x")
        return d

    def run_deploy(self, *args, **env):
        e = dict(os.environ)
        e.update({"PATH": f"{self.bin}:{os.environ['PATH']}",
                  "CALL_LOG": str(self.log), "PORT": str(self.port)})
        e.update(env)
        return subprocess.run(["bash", "scripts/deploy.sh", *args],
                              cwd=self.repo, env=e, capture_output=True, text=True)

    def calls(self):
        return self.log.read_text().splitlines() if self.log.exists() else []

    def index_of(self, needle):
        for i, line in enumerate(self.calls()):
            if needle in line:
                return i
        return -1

    def uv_calls(self):
        return [c for c in self.calls() if c.startswith("uv ")]

    def build_paths(self):
        out = []
        for c in self.calls():
            if c.startswith("arduino-cli compile"):
                parts = c.split()
                out.append(parts[parts.index("--build-path") + 1])
        return out


class ArgumentsAreCheckedFirst(DeployHarness):
    """A wrong argument must stop before any tool runs, not halfway through."""

    def test_no_argument(self):
        r = self.run_deploy()
        self.assertEqual(r.returncode, 2)
        self.assertEqual(self.calls(), [])

    def test_unknown_model(self):
        r = self.run_deploy("gpt5")
        self.assertEqual(r.returncode, 2)
        self.assertEqual(self.calls(), [])

    def test_extra_argument(self):
        r = self.run_deploy("barista", "tinystories")
        self.assertEqual(r.returncode, 2)
        self.assertEqual(self.calls(), [])

    def test_usage_names_both_models(self):
        r = self.run_deploy()
        self.assertIn("tinystories", r.stderr)
        self.assertIn("barista", r.stderr)


class MissingArtifactsStopEverything(DeployHarness):
    def test_nothing_is_generated_without_artifacts(self):
        r = self.run_deploy("barista")
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(self.calls(), [],
                         "a generator ran before the artifacts were checked")

    def test_every_missing_file_is_named(self):
        # Reported as a list, so one run tells you everything to fetch.
        self.artifacts("barista", ["model.bin"])
        r = self.run_deploy("barista")
        for name in ("tokenizer.json", "vocab.json", "layout.json"):
            self.assertIn(name, r.stderr)

    def test_barista_needs_four_files(self):
        self.artifacts("barista", ["model.bin", "tokenizer.json"])
        r = self.run_deploy("barista")
        self.assertNotEqual(r.returncode, 0)

    def test_tinystories_needs_two(self):
        self.artifacts("tinystories", ["model.bin", "tokenizer.json"])
        r = self.run_deploy("tinystories")
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_the_fetch_hint_names_the_selected_model(self):
        r = self.run_deploy("barista")
        self.assertIn("fetch_model.sh barista", r.stderr)


class ModelSelectionPicksMatchingParts(DeployHarness):
    def setUp(self):
        super().setUp()
        self.artifacts("tinystories", ["model.bin", "tokenizer.json"])
        self.artifacts("barista", ["model.bin", "tokenizer.json",
                                   "vocab.json", "layout.json"])

    def test_barista_uses_its_own_sketch_and_generators(self):
        self.assertEqual(self.run_deploy("barista").returncode, 0)
        joined = "\n".join(self.calls())
        self.assertIn("esp32_barista/tools/generate_vocab_headers.py", joined)
        self.assertIn("esp32_barista/tools/generate_tokenizer_header.py", joined)
        self.assertIn("firmware/esp32_barista", joined)
        self.assertNotIn("esp32_tinystories", joined)

    def test_tinystories_uses_its_own_sketch_and_generator(self):
        self.assertEqual(self.run_deploy("tinystories").returncode, 0)
        joined = "\n".join(self.calls())
        self.assertIn("esp32_tinystories/tools/generate_vocab.py", joined)
        self.assertIn("firmware/esp32_tinystories", joined)
        self.assertNotIn("esp32_barista", joined)

    def test_the_flashed_binary_is_the_selected_model(self):
        self.assertEqual(self.run_deploy("barista").returncode, 0)
        flash = [c for c in self.calls() if "write_flash" in c]
        self.assertEqual(len(flash), 1)
        self.assertIn("artifacts/barista/model.bin", flash[0])

    def test_artifacts_override_is_honoured(self):
        other = self.repo / "elsewhere"
        other.mkdir()
        for f in ("model.bin", "tokenizer.json", "vocab.json", "layout.json"):
            (other / f).write_text("x")
        self.assertEqual(self.run_deploy("barista", ARTIFACTS="elsewhere").returncode, 0)
        flash = [c for c in self.calls() if "write_flash" in c]
        self.assertIn("elsewhere/model.bin", flash[0])

    def test_only_the_barista_gate_runs_for_barista(self):
        self.run_deploy("barista")
        self.assertIn("verify_tokenizer.py", "\n".join(self.calls()))

    def test_no_encoder_gate_for_tinystories(self):
        self.run_deploy("tinystories")
        self.assertNotIn("verify_tokenizer.py", "\n".join(self.calls()))


class OrderProtectsTheBoard(DeployHarness):
    def setUp(self):
        super().setUp()
        self.artifacts("barista", ["model.bin", "tokenizer.json",
                                   "vocab.json", "layout.json"])

    def test_compile_precedes_both_writes(self):
        self.assertEqual(self.run_deploy("barista").returncode, 0)
        compile_at = self.index_of("arduino-cli compile")
        flash_at = self.index_of("write_flash")
        upload_at = self.index_of("arduino-cli upload")
        self.assertNotEqual(compile_at, -1)
        self.assertLess(compile_at, flash_at)
        self.assertLess(compile_at, upload_at)

    def test_model_is_written_before_firmware(self):
        self.run_deploy("barista")
        self.assertLess(self.index_of("write_flash"),
                        self.index_of("arduino-cli upload"))

    def test_gates_precede_compile(self):
        self.run_deploy("barista")
        self.assertLess(self.index_of("verify_tokenizer.py"),
                        self.index_of("arduino-cli compile"))

    def test_a_failed_compile_writes_nothing(self):
        failing = STUB.format(
            name="arduino-cli",
            body='if [ "$1" = "compile" ]; then echo "error: expected ; before }" >&2; exit 1; fi')
        (self.bin / "arduino-cli").write_text(failing)
        os.chmod(self.bin / "arduino-cli", 0o755)
        r = self.run_deploy("barista")
        self.assertNotEqual(r.returncode, 0)
        joined = "\n".join(self.calls())
        self.assertNotIn("write_flash", joined)
        self.assertNotIn("arduino-cli upload", joined)

    def test_a_failed_compile_shows_the_compiler_output(self):
        # The whole point of not piping through tail: the error survives.
        failing = STUB.format(
            name="arduino-cli",
            body=('if [ "$1" = "compile" ]; then '
                  'for i in 1 2 3 4 5 6 7 8; do echo "error: line $i" >&2; done; exit 1; fi'))
        (self.bin / "arduino-cli").write_text(failing)
        os.chmod(self.bin / "arduino-cli", 0o755)
        r = self.run_deploy("barista")
        for i in range(1, 9):
            self.assertIn(f"error: line {i}", r.stderr,
                          "compiler output was truncated")

    def test_a_failed_gate_stops_before_compiling(self):
        failing = STUB.format(name="uv", body='echo "FAIL: mismatch" >&2; exit 1')
        (self.bin / "uv").write_text(failing)
        os.chmod(self.bin / "uv", 0o755)
        r = self.run_deploy("barista")
        self.assertNotEqual(r.returncode, 0)
        self.assertNotIn("arduino-cli compile", "\n".join(self.calls()))


class ToolsRunOutsideTheProjectEnvironment(DeployHarness):
    """Deploying must not synchronise or disturb the research environment, and
    must not let a new release of the tokenizer change what it builds."""

    def setUp(self):
        super().setUp()
        self.artifacts("tinystories", ["model.bin", "tokenizer.json"])
        self.artifacts("barista", ["model.bin", "tokenizer.json",
                                   "vocab.json", "layout.json"])

    def test_no_uv_call_uses_the_project_environment(self):
        for model, expected in (("barista", 3), ("tinystories", 2)):
            with self.subTest(model=model):
                self.log.unlink(missing_ok=True)
                self.assertEqual(self.run_deploy(model).returncode, 0)
                calls = self.uv_calls()
                self.assertEqual(len(calls), expected)
                for c in calls:
                    self.assertIn("--no-project", c)

    def test_every_added_dependency_is_pinned(self):
        for model in ("barista", "tinystories"):
            with self.subTest(model=model):
                self.log.unlink(missing_ok=True)
                self.run_deploy(model)
                withs = [c for c in self.uv_calls() if "--with" in c]
                self.assertEqual(len(withs), 1)
                self.assertIn("tokenizers==0.23.1", withs[0])

    def test_nothing_asks_for_torch(self):
        self.run_deploy("barista")
        self.assertNotIn("torch", "\n".join(self.uv_calls()))


class RunsDoNotShareAWorkspace(DeployHarness):
    def setUp(self):
        super().setUp()
        self.artifacts("barista", ["model.bin", "tokenizer.json",
                                   "vocab.json", "layout.json"])

    def test_each_run_builds_somewhere_of_its_own(self):
        self.assertEqual(self.run_deploy("barista").returncode, 0)
        self.assertEqual(self.run_deploy("barista").returncode, 0)
        paths = self.build_paths()
        self.assertEqual(len(paths), 2)
        self.assertNotEqual(paths[0], paths[1],
                            "two runs compiled into the same directory")

    def test_the_workspace_is_removed_afterwards(self):
        self.run_deploy("barista")
        for p in self.build_paths():
            self.assertFalse(Path(p).exists(), f"{p} outlived the run")


class NoBoardNoRun(DeployHarness):
    def test_a_missing_port_stops_before_generating(self):
        # Stub auto-detection so the test is independent of connected hardware.
        stub_ls = self.bin / "ls"
        stub_ls.write_text("#!/bin/sh\nexit 1\n")
        os.chmod(stub_ls, 0o755)
        self.artifacts("barista", ["model.bin", "tokenizer.json",
                                   "vocab.json", "layout.json"])
        r = self.run_deploy("barista", PORT="")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no /dev/cu.usbmodem*", r.stderr)
        self.assertEqual(self.calls(), [])


if __name__ == "__main__":
    unittest.main()
