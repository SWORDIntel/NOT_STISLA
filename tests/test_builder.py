#!/usr/bin/env python3
"""
Unit tests for KEYSTONE builder.py — validation, CPU detection, UI helpers.

Run: python3 tests/test_builder.py
"""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Import builder module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import builder


class TestValidateMarch(unittest.TestCase):
    """Test --arch validation."""

    def test_valid_known_marches(self):
        for m in ("native", "sandybridge", "haswell", "alderlake",
                   "skylake-avx512", "x86-64", "x86-64-v2", "x86-64-v4"):
            self.assertEqual(builder.validate_march(m), m)

    def test_valid_format_unknown(self):
        # Valid format but not in known list — should warn but return
        result = builder.validate_march("znver4")
        self.assertEqual(result, "znver4")

    def test_rejects_shell_metacharacters(self):
        with self.assertRaises(SystemExit):
            builder.validate_march("native; rm -rf /")

    def test_rejects_spaces(self):
        with self.assertRaises(SystemExit):
            builder.validate_march("native -O0")

    def test_rejects_empty(self):
        with self.assertRaises(SystemExit):
            builder.validate_march("")

    def test_rejects_uppercase(self):
        with self.assertRaises(SystemExit):
            builder.validate_march("Native")

    def test_allows_dots_and_dashes(self):
        self.assertEqual(builder.validate_march("x86-64-v4"), "x86-64-v4")
        self.assertEqual(builder.validate_march("skylake-avx512"), "skylake-avx512")


class TestValidateAlias(unittest.TestCase):
    """Test shell alias name validation."""

    def test_valid_names(self):
        for name in ("KEYSTONE_DB", "MY_VAR", "_private", "A", "VAR_123"):
            self.assertEqual(builder.validate_alias(name), name)

    def test_rejects_semicolon(self):
        with self.assertRaises(SystemExit):
            builder.validate_alias("FOO; rm /")

    def test_rejects_starts_with_digit(self):
        with self.assertRaises(SystemExit):
            builder.validate_alias("1VAR")

    def test_rejects_spaces(self):
        with self.assertRaises(SystemExit):
            builder.validate_alias("MY VAR")

    def test_rejects_dollar(self):
        with self.assertRaises(SystemExit):
            builder.validate_alias("$(whoami)")

    def test_rejects_empty(self):
        with self.assertRaises(SystemExit):
            builder.validate_alias("")


class TestVisibleLen(unittest.TestCase):
    """Test ANSI-stripped visible length calculation."""

    def test_plain_string(self):
        self.assertEqual(builder.visible_len("hello"), 5)

    def test_ansi_stripped(self):
        s = builder.c("hello", builder.RED)
        self.assertEqual(builder.visible_len(s), 5)

    def test_empty(self):
        self.assertEqual(builder.visible_len(""), 0)

    def test_wide_chars(self):
        # Box drawing chars are not wide (Na), so count as 1
        self.assertEqual(builder.visible_len("─"), 1)

    def test_mixed_ansi_and_text(self):
        s = f"{builder.RED}error{builder.RST}: {builder.GREEN}ok{builder.RST}"
        self.assertEqual(builder.visible_len(s), len("error: ok"))


class TestPadRight(unittest.TestCase):
    """Test right-padding to visible width."""

    def test_short_string(self):
        result = builder.pad_right("hi", 10)
        self.assertEqual(builder.visible_len(result), 10)
        self.assertTrue(result.startswith("hi"))

    def test_exact_width(self):
        result = builder.pad_right("hello", 5)
        self.assertEqual(result, "hello")

    def test_oversized_string(self):
        result = builder.pad_right("hello world", 5)
        self.assertEqual(result, "hello world")

    def test_empty_string(self):
        result = builder.pad_right("", 5)
        self.assertEqual(result, "     ")


class TestDetectCpu(unittest.TestCase):
    """Test CPU detection returns expected structure."""

    def test_returns_dict_with_required_keys(self):
        feat = builder.detect_cpu()
        required = {"arch", "model", "sse42", "avx", "avx2", "fma",
                    "avx512", "amx", "vnni", "hybrid",
                    "hybrid_pcores", "hybrid_ecores"}
        for key in required:
            self.assertIn(key, feat, f"Missing key: {key}")

    def test_arch_is_string(self):
        feat = builder.detect_cpu()
        self.assertIsInstance(feat["arch"], str)
        self.assertTrue(len(feat["arch"]) > 0)

    def test_model_is_string(self):
        feat = builder.detect_cpu()
        self.assertIsInstance(feat["model"], str)

    def test_boolean_fields(self):
        feat = builder.detect_cpu()
        for key in ("sse42", "avx", "avx2", "fma", "avx512", "amx", "hybrid"):
            self.assertIsInstance(feat[key], bool, f"{key} should be bool")

    def test_lists_for_hybrid(self):
        feat = builder.detect_cpu()
        self.assertIsInstance(feat["hybrid_pcores"], list)
        self.assertIsInstance(feat["hybrid_ecores"], list)

    def test_avx512_requires_foundation(self):
        feat = builder.detect_cpu()
        if feat["avx512"]:
            self.assertTrue(feat["avx512f"])
            self.assertTrue(feat["avx512dq"])
            self.assertTrue(feat["avx512bw"])
            self.assertTrue(feat["avx512vl"])


class TestArchLabel(unittest.TestCase):
    """Test ISA label formatting."""

    def test_scalar_fallback(self):
        feat = {"avx512": False, "amx": False, "avx_vnni": False,
                "avx2": False, "avx": False, "sse42": False, "fma": False}
        self.assertEqual(builder.arch_label(feat), "scalar")

    def test_sse42(self):
        feat = {"avx512": False, "amx": False, "avx_vnni": False,
                "avx2": False, "avx": False, "sse42": True, "fma": False}
        self.assertEqual(builder.arch_label(feat), "SSE4.2")

    def test_avx1(self):
        feat = {"avx512": False, "amx": False, "avx_vnni": False,
                "avx2": False, "avx": True, "sse42": True, "fma": False}
        self.assertEqual(builder.arch_label(feat), "AVX1")

    def test_avx2_with_fma(self):
        feat = {"avx512": False, "amx": False, "avx_vnni": False,
                "avx2": True, "avx": True, "sse42": True, "fma": True}
        label = builder.arch_label(feat)
        self.assertIn("AVX2", label)
        self.assertIn("FMA", label)

    def test_avx512_includes_subsets(self):
        feat = {"avx512": True, "avx512_bf16": True, "avx512_fp16": False,
                "avx512_vnni": True, "avx512_vbmi": False, "avx512_ifma": False,
                "amx": False, "avx_vnni": False, "avx2": True, "avx": True,
                "sse42": True, "fma": True}
        label = builder.arch_label(feat)
        self.assertIn("AVX-512", label)
        self.assertIn("BF16", label)
        self.assertIn("VNNI", label)


class TestSafeInput(unittest.TestCase):
    """Test EOF-safe input wrapper."""

    def test_returns_default_on_eof(self):
        with patch("builtins.input", side_effect=EOFError):
            result = builder.safe_input("prompt", "default_val")
            self.assertEqual(result, "default_val")

    def test_returns_input_on_success(self):
        with patch("builtins.input", return_value="user_text"):
            result = builder.safe_input("prompt", "default_val")
            self.assertEqual(result, "user_text")


class TestBoxAlignment(unittest.TestCase):
    """Test that box functions produce consistent widths."""

    def test_banner_width(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            builder.banner("TEST")
        lines = [l for l in buf.getvalue().split("\n") if l.strip()]
        for line in lines:
            clean = builder._ANSI_RE.sub("", line)
            self.assertEqual(len(clean), builder.WIDTH + 2,
                             f"Banner line width mismatch: {len(clean)} != {builder.WIDTH + 2}")

    def test_success_box_width(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            builder.success_box("Test message")
        lines = [l for l in buf.getvalue().split("\n") if l.strip()]
        for line in lines:
            clean = builder._ANSI_RE.sub("", line)
            self.assertEqual(len(clean), builder.WIDTH + 2,
                             f"Success box width mismatch: {len(clean)} != {builder.WIDTH + 2}")

    def test_warning_box_width(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            builder.warning_box(["Line 1", "Line 2"])
        lines = [l for l in buf.getvalue().split("\n") if l.strip()]
        for line in lines:
            clean = builder._ANSI_RE.sub("", line)
            self.assertEqual(len(clean), builder.WIDTH + 2,
                             f"Warning box width mismatch: {len(clean)} != {builder.WIDTH + 2}")


class TestBuildTargets(unittest.TestCase):
    """Test build target definitions."""

    def test_all_targets_have_three_elements(self):
        for key, val in builder.BUILD_TARGETS.items():
            self.assertEqual(len(val), 2, f"Target {key} should have (id, desc)")

    def test_arch_targets_have_three_elements(self):
        for key, val in builder.ARCH_TARGETS.items():
            self.assertEqual(len(val), 3, f"Arch target {key} should have (id, desc, march)")

    def test_native_is_option_1(self):
        self.assertEqual(builder.ARCH_TARGETS["1"][0], "native")
        self.assertIsNone(builder.ARCH_TARGETS["1"][2])

    def test_submenu_marker(self):
        self.assertEqual(builder.ARCH_TARGETS["5"][2], "SUBMENU")


if __name__ == "__main__":
    unittest.main()
