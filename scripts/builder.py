#!/usr/bin/env python3
"""
KEYSTONE Builder — architecture-aware build orchestrator with dep checking.

Auto-detects CPU ISA, offers target override (cross-compile), checks all
system deps (libarchive, libzstd, gfortran), then compiles.

Theme: SWORD cyber-dark (#08080a bg, #e50000 accent, Share Tech Mono spirit).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def visible_len(s: str) -> int:
    """Length of string excluding ANSI escape codes, counting wide chars as 2."""
    clean = _ANSI_RE.sub("", s)
    w = 0
    for ch in clean:
        ea = unicodedata.east_asian_width(ch)
        w += 2 if ea in ("W", "F") else 1
    return w


def pad_right(s: str, width: int) -> str:
    """Pad string on the right so its visible width equals width."""
    vl = visible_len(s)
    return s + " " * max(0, width - vl)

# ── SWORD cyber-dark ANSI palette ──────────────────────────────────────
BG     = "\033[48;5;233m"   # #08080a near-black
RST    = "\033[0m"
RED    = "\033[38;5;196m"   # #e50000 SWORD red
CYAN   = "\033[36m"          # T420 label
ORANGE = "\033[38;5;208m"   # T320 label
BOLD   = "\033[1m"
DIM    = "\033[2m"
WHITE  = "\033[97m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
GRAY   = "\033[38;5;240m"

WIDTH = 63


def c(text: str, *colors: str) -> str:
    return f"{''.join(colors)}{text}{RST}"


def banner(title: str) -> None:
    inner = WIDTH - 2
    top    = f"╭{'─' * inner}╮"
    bottom = f"╰{'─' * inner}╯"
    vl = visible_len(title)
    pad = max(0, inner - 2 - vl)
    left = pad // 2
    right = pad - left
    print(f"\n  {c(top, RED)}")
    print(f"  {c('│', RED)} {' ' * left}{c(title, RED, BOLD)}{' ' * right} {c('│', RED)}")
    print(f"  {c(bottom, RED)}")


def section(title: str) -> None:
    print(f"\n  {c('◆', RED)} {c(title, RED, BOLD)}")
    print(f"  {c('─' * (WIDTH - 2), GRAY)}")


def warning_box(lines: list[str]) -> None:
    inner = WIDTH - 4
    top    = f"╭{'─' * inner}╮"
    bottom = f"╰{'─' * inner}╯"
    print(f"\n  {c(top, YELLOW, BOLD)}")
    for line in lines:
        print(f"  {c('│', YELLOW, BOLD)} {pad_right(line, inner - 2)} {c('│', YELLOW, BOLD)}")
    print(f"  {c(bottom, YELLOW, BOLD)}")


def info(msg: str) -> None:
    print(f"  {c('●', CYAN)} {msg}")


def ok(msg: str) -> None:
    print(f"  {c('✓', GREEN)} {msg}")


def warn(msg: str) -> None:
    print(f"  {c('⚠', YELLOW)} {msg}")


def fail(msg: str) -> None:
    print(f"  {c('✗', RED)} {msg}")


def run(cmd: list[str] | str, cwd: Path | None = None, check: bool = True,
        env: dict | None = None, shell: bool = False) -> int:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    if shell:
        assert isinstance(cmd, str)
        return subprocess.run(cmd, cwd=cwd, env=full_env, shell=True).returncode
    assert isinstance(cmd, list)
    rc = subprocess.run(cmd, cwd=cwd, env=full_env).returncode
    if check and rc != 0:
        fail(f"Command failed (exit {rc}): {' '.join(cmd)}")
        sys.exit(rc)
    return rc


# ── Architecture detection ─────────────────────────────────────────────

def detect_cpu() -> dict:
    """Detect CPU architecture and SIMD features via /proc/cpuinfo."""
    flags: list[str] = []
    arch = "x86_64"
    model = "unknown"

    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags") and not flags:
                    flags = line.split(":", 1)[1].split()
                if line.startswith("model name") and model == "unknown":
                    model = line.split(":", 1)[1].strip()
    except FileNotFoundError:
        pass

    if not flags:
        arch = os.uname().machine
        try:
            out = subprocess.check_output(["lscpu"], text=True)
            for line in out.splitlines():
                if line.startswith("Flags:"):
                    flags = line.split(":", 1)[1].split()
                if line.startswith("Model name:"):
                    model = line.split(":", 1)[1].strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

    feat = {
        "arch": arch,
        "model": model,
        "sse42": "sse4_2" in flags,
        "avx": "avx" in flags,
        "avx2": "avx2" in flags,
        "avx512f": "avx512f" in flags,
        "avx512dq": "avx512dq" in flags,
        "avx512bw": "avx512bw" in flags,
        "avx512vl": "avx512vl" in flags,
        "aesni": "aes" in flags,
        "fma": "fma" in flags,
        "amx": "amx_tile" in flags,
        "vnni": "avx512_vnni" in flags,
        "f16c": "f16c" in flags,
    }
    feat["avx512"] = all(feat[k] for k in ("avx512f", "avx512dq", "avx512bw", "avx512vl"))
    return feat


def arch_label(feat: dict) -> str:
    if feat["avx512"]:
        return "AVX-512"
    if feat["avx2"]:
        return "AVX2"
    if feat["avx"]:
        return "AVX1"
    if feat["sse42"]:
        return "SSE4.2"
    return "scalar"


# ── Target menu (cross-compile override) ───────────────────────────────

ARCH_TARGETS = {
    "1": ("native",  "Auto-detect (march=native)", None),
    "2": ("sandy",   "Sandy Bridge (-march=sandybridge)", "sandybridge"),
    "3": ("haswell", "Haswell AVX2 (-march=haswell)", "haswell"),
    "4": ("skylake", "Skylake AVX-512 (-march=skylake-avx512)", "skylake-avx512"),
    "5": ("generic", "Generic x86-64-v2 (-march=x86-64-v2)", "x86-64-v2"),
    "6": ("scalar",  "Scalar only (-march=x86-64)", "x86-64"),
}

BUILD_TARGETS = {
    "1": ("all",        "Full build (lib + tests + benchmarks)"),
    "2": ("lib",        "Library only (libkeystone.so)"),
    "3": ("tests",      "Test binaries"),
    "4": ("benchmarks", "Benchmark binaries"),
}


def menu(title: str, options: dict, default: str) -> str:
    section(title)
    for key in sorted(options):
        desc = options[key][1]
        marker = f" {c('▸', CYAN)} " if key == default else "   "
        print(f"  {marker}{c(f'[{key}]', BOLD)} {desc}")
    choice = input(f"\n  {c('◆', RED)} {c('Select', WHITE)} [{default}]: ").strip() or default
    if choice not in options:
        fail(f"Invalid choice: {choice}")
        sys.exit(1)
    return choice


# ── Dependency checking ────────────────────────────────────────────────

APT_DEPS = [
    ("gcc", "gcc"),
    ("gfortran", "gfortran"),
    ("libarchive-dev", "libarchive-dev"),
    ("libzstd-dev", "libzstd-dev"),
]

OPTIONAL_DEPS = [
    ("libqihse", "QIHSE library (libqihse.so)", "/opt/qihse/lib/libqihse.so"),
]


def check_apt_deps() -> list[str]:
    """Return list of missing apt packages."""
    missing = []
    for pkg, _ in APT_DEPS:
        r = subprocess.run(["dpkg", "-s", pkg], capture_output=True)
        if r.returncode != 0:
            missing.append(pkg)
    return missing


def install_apt_deps(missing: list[str]) -> None:
    if not missing:
        ok("All system dependencies present")
        return
    warn(f"Missing system packages: {', '.join(missing)}")
    if input(f"  {c('Install via apt?', WHITE)} [Y/n] ").strip().lower() not in ("n", "no"):
        run(["sudo", "apt-get", "install", "-y"] + missing)


def check_optional_deps() -> None:
    for name, desc, path in OPTIONAL_DEPS:
        if Path(path).exists():
            ok(f"{name} found at {path}")
        else:
            warn(f"{name} not found — {desc} (build will skip QIHSE bridge)")


def provision_deps() -> None:
    """Check all deps before building."""
    banner("DEPENDENCY CHECK")
    missing = check_apt_deps()
    install_apt_deps(missing)
    check_optional_deps()


# ── Build ──────────────────────────────────────────────────────────────

def build(march: str | None, target: str, clean: bool, jobs: int) -> None:
    banner("COMPILING")
    make_args = ["make", f"-j{jobs}"]

    if clean:
        info("Cleaning previous build...")
        run(["make", "clean"], cwd=ROOT, check=False)

    march_val = march if march else "native"

    make_args += [
        f"MARCH={march_val}",
        target,
    ]

    info(f"Target: {c(target, CYAN)}  march={c(march or 'native', CYAN)}  jobs={c(str(jobs), CYAN)}")
    info(f"Command: {' '.join(make_args)}")
    print()
    rc = run(make_args, cwd=ROOT, check=False)
    print()
    if rc == 0:
        ok(f"Build successful — {target}")
    else:
        fail(f"Build failed (exit {rc})")
        sys.exit(rc)


def install_opt() -> None:
    """Install built artifacts to /opt/keystone."""
    banner("INSTALL → /opt/keystone")
    dest = Path("/opt/keystone")
    if not dest.exists():
        warn("/opt/keystone does not exist, creating (needs sudo)...")
        run(["sudo", "mkdir", "-p", str(dest / "bin"), str(dest / "lib"), str(dest / "include" / "keystone")])
        run(["sudo", "chown", "-R", f"{os.getuid()}:{os.getgid()}", str(dest)])

    lib = ROOT / "libkeystone.so"
    fortran_lib = ROOT / "fortran" / "libkeystone_batch.so"

    if lib.exists():
        shutil.copy2(lib, dest / "lib" / "libkeystone.so")
        ok(f"libkeystone.so → {dest}/lib/")
    if fortran_lib.exists():
        shutil.copy2(fortran_lib, dest / "lib" / "libkeystone_batch.so")
        ok(f"libkeystone_batch.so → {dest}/lib/")

    # Binaries
    bin_count = 0
    for b in (ROOT / "bin").glob("test_*"):
        shutil.copy2(b, dest / "bin" / b.name)
        bin_count += 1
    for b in (ROOT / "benchmarks").glob("*_benchmark"):
        if b.is_file() and os.access(b, os.X_OK):
            shutil.copy2(b, dest / "bin" / b.name)
            bin_count += 1
    ok(f"binaries → {dest}/bin/ ({bin_count} files)")

    # Headers
    inc_src = ROOT / "include"
    inc_dst = dest / "include" / "keystone"
    inc_dst.mkdir(parents=True, exist_ok=True)
    for h in inc_src.glob("*.h"):
        shutil.copy2(h, inc_dst / h.name)
    ok(f"headers → {dest}/include/keystone/ ({len(list(inc_dst.glob('*.h')))} files)")

    # Symlink
    link = dest / "libkeystone.so"
    if not link.exists() or link.is_symlink():
        try:
            link.unlink(missing_ok=True)
        except OSError:
            pass
        link.symlink_to("lib/libkeystone.so")
        ok(f"symlink → {dest}/libkeystone.so")


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    banner("K E Y S T O N E")
    feat = detect_cpu()

    section("ARCHITECTURE DETECTION")
    info(f"CPU:    {c(feat['model'], WHITE)}")
    info(f"Arch:   {c(feat['arch'], CYAN)}  ISA: {c(arch_label(feat), CYAN)}")
    feats = [k.upper() for k in ("sse42", "avx", "avx2", "avx512", "aesni", "fma", "amx", "vnni", "f16c") if feat[k]]
    info(f"Flags:  {c(' '.join(feats), DIM)}")

    # Check deps
    provision_deps()

    # Target menu
    arch_choice = menu("Build target (architecture)", ARCH_TARGETS, "1")
    _, arch_desc, march_override = ARCH_TARGETS[arch_choice]

    # Cross-compile warning
    if arch_choice != "1":
        warning_box([
            c("⚠  CROSS-COMPILATION WARNING", YELLOW, BOLD),
            "",
            f"  You selected: {arch_desc}",
            f"  Your CPU ISA:  {arch_label(feat)}",
            "",
            "  Building for a different architecture than this",
            "  CPU means the binaries may NOT run on this PC.",
            "  Use this only when deploying to another machine.",
            "",
            "  If unsure, select [1] Auto-detect (march=native).",
        ])
        if input(f"  {c('◆', RED)} {c('Proceed anyway?', WHITE)} [y/N] ").strip().lower() not in ("y", "yes"):
            warn("Aborted — no changes made.")
            sys.exit(0)

    build_choice = menu("Build target (make goal)", BUILD_TARGETS, "1")
    _, build_desc = BUILD_TARGETS[build_choice]

    clean = input(f"\n  {c('◆', RED)} {c('Clean before build?', WHITE)} [y/N] ").strip().lower() in ("y", "yes")
    jobs = os.cpu_count() or 4

    march = march_override

    # Summary
    section("BUILD SUMMARY")
    info(f"Architecture:  {c(arch_desc, CYAN)}")
    info(f"Build goal:    {c(build_desc, CYAN)}")
    info(f"Clean:         {c('yes' if clean else 'no', CYAN)}   Jobs: {c(str(jobs), CYAN)}")
    info(f"Output:        {c(str(ROOT / 'libkeystone.so'), DIM)}")

    build(march, BUILD_TARGETS[build_choice][0], clean, jobs)

    # Offer install
    if input(f"\n  {c('◆', RED)} {c('Install to /opt/keystone?', WHITE)} [Y/n] ").strip().lower() not in ("n", "no"):
        install_opt()

    done_line = c("Done.", GREEN, BOLD) + "  " + c("KEYSTONE build complete.", DIM)
    inner_w = WIDTH - 4
    print(f"\n  {c('╭' + '─' * (WIDTH - 2) + '╮', GREEN)}")
    print(f"  {c('│', GREEN)} {pad_right(done_line, inner_w)} {c('│', GREEN)}")
    print(f"  {c('╰' + '─' * (WIDTH - 2) + '╯', GREEN)}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n  {c('Aborted.', RED)}")
        sys.exit(130)
