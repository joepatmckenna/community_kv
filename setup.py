"""Build the community_kv package + all CUDA extensions in a single
``pip install -e .`` invocation.

Orchestration (idempotent — re-running only redoes what's needed):

  1. Initialize the ``third_party/flash-attention`` git submodule.
  2. Apply ``community_kv/attention/flash-attention.patch`` to the
     submodule (skip if already applied).
  3. Restore the patched flash-attention hopper artifacts from the
     ``COMMUNITY_KV_BUILD_CACHE`` directory if present and importable on
     this machine; otherwise build via the upstream ``hopper/setup.py``
     and populate the cache.
  4. Build the Leiden CUDA extension (or restore from cache and verify
     it loads).

Cache validity is determined by an "does it import?" probe: cached files
are restored into place, then imported in a clean subprocess. Any
failure (ABI mismatch, broken .so, missing symbol) triggers a rebuild.

Disable any of the orchestration steps via env vars:
    COMMUNITY_KV_SKIP_SUBMODULE=1
    COMMUNITY_KV_SKIP_PATCH=1
    COMMUNITY_KV_SKIP_FUSED_ATTN_FWD_TOPK_BUILD=1
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).parent.resolve()
FA_DIR = ROOT / "third_party" / "flash-attention"
PATCH = ROOT / "community_kv" / "attention" / "flash-attention.patch"
LEIDEN_SRC_REL = "community_kv/graph/_leiden/_csrc"
LEIDEN_SO_NAME = "_community_kv_leiden.cpython-{abi}-{machine}-linux-gnu.so"
LEIDEN_DEST_DIR = ROOT / "community_kv" / "graph" / "_leiden"


def _log(msg: str) -> None:
    print(f"[community_kv setup] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Cache helpers — file ops + subprocess import probe.
# Cache layout:
#     {COMMUNITY_KV_BUILD_CACHE}/
#         fused_attn_fwd_topk/{flash_attn_interface.py, flash_attn_3/...}
#         leiden/_community_kv_leiden.cpython-*-linux-gnu.so
# --------------------------------------------------------------------------- #


def _cache_dir() -> Path | None:
    """Resolved build-cache dir.

    Defaults to ``<repo root>/.build`` so subsequent installs on the same
    checkout reuse compiled artifacts without any extra configuration.
    Override with ``COMMUNITY_KV_BUILD_CACHE=<path>`` (e.g., point at a
    shared directory across checkouts), or disable entirely with
    ``COMMUNITY_KV_BUILD_CACHE=0`` / ``COMMUNITY_KV_BUILD_CACHE=``.
    """
    raw = os.environ.get("COMMUNITY_KV_BUILD_CACHE")
    if raw is None:
        p = ROOT / ".build"
    else:
        if raw in ("", "0"):
            return None
        p = Path(raw).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _copy_into(src: Path, dst: Path) -> None:
    """Copy src (file or dir) to dst, replacing any existing dst."""
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _probe_import(snippet: str) -> bool:
    """Run a tiny Python snippet in a clean subprocess; return True on success."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    res = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        env=env,
    )
    return res.returncode == 0


# --------------------------------------------------------------------------- #
# FA3 hopper steps
# --------------------------------------------------------------------------- #


def _fused_attn_fwd_topk_pth_path() -> Path:
    """site-packages .pth file pointing at the hopper dir so
    ``import flash_attn_interface`` works without ``pip install``."""
    import site

    return Path(site.getsitepackages()[0]) / "community_kv_fused_attn_fwd_topk.pth"


def _write_fused_attn_fwd_topk_pth() -> None:
    pth = _fused_attn_fwd_topk_pth_path()
    target = str(FA_DIR / "hopper")
    pth.write_text(target + "\n")
    _log(f"wrote {pth} -> {target}")


def ensure_submodule() -> None:
    if os.environ.get("COMMUNITY_KV_SKIP_SUBMODULE") == "1":
        _log("COMMUNITY_KV_SKIP_SUBMODULE=1 -> skip submodule init")
        return
    if (FA_DIR / "hopper").exists():
        _log(f"submodule already populated at {FA_DIR}")
        return
    if not (ROOT / ".git").exists():
        raise RuntimeError(
            f"third_party/flash-attention is empty and {ROOT} isn't a git "
            f"checkout — can't auto-init the submodule. Either clone with "
            f"--recurse-submodules, or symlink an existing flash-attention "
            f"checkout into {FA_DIR}, or set COMMUNITY_KV_SKIP_SUBMODULE=1."
        )
    _log("git submodule update --init --recursive third_party/flash-attention")
    subprocess.run(
        ["git", "submodule", "update", "--init", "--recursive", str(FA_DIR.relative_to(ROOT))],
        cwd=str(ROOT),
        check=True,
    )


def ensure_patch_applied() -> None:
    if os.environ.get("COMMUNITY_KV_SKIP_PATCH") == "1":
        _log("COMMUNITY_KV_SKIP_PATCH=1 -> skip patch apply")
        return
    if not PATCH.exists():
        raise RuntimeError(f"missing patch file: {PATCH}")
    rev = subprocess.run(
        ["git", "apply", "--check", "--reverse", str(PATCH)],
        cwd=str(FA_DIR),
        capture_output=True,
    )
    if rev.returncode == 0:
        _log("flash-attention.patch already applied")
        return
    _log("applying flash-attention.patch ...")
    subprocess.run(["git", "apply", str(PATCH)], cwd=str(FA_DIR), check=True)


def _restore_fused_attn_fwd_topk_from_cache() -> bool:
    """Restore FA3 artifacts from cache and verify by import. Returns True
    iff the restored build is usable on this machine."""
    cache = _cache_dir()
    if cache is None:
        return False
    slot = cache / "fused_attn_fwd_topk"
    iface = slot / "flash_attn_interface.py"
    so_glob = (
        list((slot / "flash_attn_3").glob("_C.*.so")) if (slot / "flash_attn_3").is_dir() else []
    )
    if not (iface.exists() and so_glob):
        return False

    _log(f"restoring flash-attention hopper artifacts from cache: {slot}")
    hopper = FA_DIR / "hopper"
    _copy_into(slot / "flash_attn_3", hopper / "flash_attn_3")
    _copy_into(iface, hopper / "flash_attn_interface.py")
    _write_fused_attn_fwd_topk_pth()

    if _probe_import("import flash_attn_interface"):
        _log("cache validated: flash_attn_interface imports cleanly")
        return True
    _log("cache miss: cached artifacts failed to import on this machine")
    # Tear down the .pth so a downstream reinstall re-evaluates cleanly.
    pth = _fused_attn_fwd_topk_pth_path()
    if pth.exists():
        pth.unlink()
    return False


def _save_fused_attn_fwd_topk_to_cache() -> None:
    cache = _cache_dir()
    if cache is None:
        return
    slot = cache / "fused_attn_fwd_topk"
    iface = FA_DIR / "hopper" / "flash_attn_interface.py"
    so_dir = FA_DIR / "hopper" / "flash_attn_3"
    if not iface.exists() or not so_dir.exists():
        _log(f"flash-attention artifacts missing at {FA_DIR / 'hopper'}; skipping cache save")
        return
    _log(f"saving flash-attention hopper artifacts to cache: {slot}")
    slot.mkdir(parents=True, exist_ok=True)
    _copy_into(iface, slot / "flash_attn_interface.py")
    _copy_into(so_dir, slot / "flash_attn_3")


def ensure_fused_attn_fwd_topk_built() -> None:
    if os.environ.get("COMMUNITY_KV_SKIP_FUSED_ATTN_FWD_TOPK_BUILD") == "1":
        _log("COMMUNITY_KV_SKIP_FUSED_ATTN_FWD_TOPK_BUILD=1 -> skip flash-attention build probe")
        return
    if _probe_import("import flash_attn_interface"):
        _log("flash_attn_interface already importable -> skip flash-attention build")
        # Opportunistic save: if the cache is empty but artifacts exist
        # in the hopper dir, populate it for the next install.
        cache = _cache_dir()
        if cache is not None and not (cache / "fused_attn_fwd_topk").exists():
            _save_fused_attn_fwd_topk_to_cache()
        return

    if _restore_fused_attn_fwd_topk_from_cache():
        return

    _log("building patched flash-attention hopper extension (slow step)")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "-e",
            str(FA_DIR / "hopper"),
        ],
        check=True,
    )
    _save_fused_attn_fwd_topk_to_cache()


# --------------------------------------------------------------------------- #
# Leiden steps — same shape: restore + probe; build + save on miss.
# --------------------------------------------------------------------------- #


def _leiden_so_name() -> str:
    abi = f"{sys.version_info.major}{sys.version_info.minor}"
    import platform as _p

    return LEIDEN_SO_NAME.format(abi=abi, machine=_p.machine())


def _leiden_probe_snippet(so_path: Path) -> str:
    """Standalone snippet: load the .so via importlib.util.spec_from_file_location.
    Doesn't require the community_kv package to be installed yet."""
    return (
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('_t', r'{so_path}')\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
    )


def _restore_leiden_from_cache() -> bool:
    cache = _cache_dir()
    if cache is None:
        return False
    slot = cache / "leiden"
    so_name = _leiden_so_name()
    cached_so = slot / so_name
    if not cached_so.exists():
        return False

    _log(f"restoring Leiden .so from cache: {cached_so}")
    LEIDEN_DEST_DIR.mkdir(parents=True, exist_ok=True)
    target_so = LEIDEN_DEST_DIR / so_name
    _copy_into(cached_so, target_so)

    if _probe_import(_leiden_probe_snippet(target_so)):
        _log("cache validated: Leiden .so loads cleanly")
        return True
    _log("cache miss: cached Leiden .so failed to load on this machine")
    target_so.unlink(missing_ok=True)
    return False


def _save_leiden_to_cache() -> None:
    cache = _cache_dir()
    if cache is None:
        return
    so_name = _leiden_so_name()
    src = LEIDEN_DEST_DIR / so_name
    if not src.exists():
        _log(f"Leiden .so not found at {src}; skipping cache save")
        return
    slot = cache / "leiden"
    slot.mkdir(parents=True, exist_ok=True)
    _log(f"saving Leiden .so to cache: {slot / so_name}")
    _copy_into(src, slot / so_name)


# --------------------------------------------------------------------------- #
# CUDA extension + custom build_ext
# --------------------------------------------------------------------------- #


class CommunityKVBuildExt(BuildExtension):
    """Run submodule + patch + flash-attention build before the Leiden build.

    Leiden also tries the cache: a hit (validated via load probe) skips the
    CUDA build entirely; otherwise build runs then saves the .so.
    """

    def run(self):
        cache = _cache_dir()
        _log(f"build cache: {'enabled at ' + str(cache) if cache else 'disabled'}")
        ensure_submodule()
        ensure_patch_applied()
        ensure_fused_attn_fwd_topk_built()

        if _restore_leiden_from_cache():
            _log("Leiden restored from cache -> skipping CUDA build")
            return

        super().run()
        _save_leiden_to_cache()


ext_modules = [
    CUDAExtension(
        name="community_kv.graph._leiden._community_kv_leiden",
        sources=[
            f"{LEIDEN_SRC_REL}/bindings.cpp",
            f"{LEIDEN_SRC_REL}/leiden.cu",
            f"{LEIDEN_SRC_REL}/csr.cu",
            f"{LEIDEN_SRC_REL}/local_moving.cu",
            f"{LEIDEN_SRC_REL}/helpers.cu",
            f"{LEIDEN_SRC_REL}/aggregate.cu",
            f"{LEIDEN_SRC_REL}/refinement.cu",
        ],
        include_dirs=[LEIDEN_SRC_REL],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": [
                "-O3",
                "-std=c++17",
                "--use_fast_math",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "-gencode=arch=compute_90,code=sm_90",
            ],
        },
    ),
]


setup(ext_modules=ext_modules, cmdclass={"build_ext": CommunityKVBuildExt})
