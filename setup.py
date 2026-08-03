from __future__ import annotations

import hashlib
import os
import shutil
import site
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup

ROOT = Path(__file__).resolve().parent
FLASH_ATTENTION_REPOSITORY = "https://github.com/Dao-AILab/flash-attention.git"
FLASH_ATTENTION_COMMIT = "060c9188beec3a8b62b33a3bfa6d5d2d44975fab"
FLASH_ATTENTION_PATCH = ROOT / "third_party/flash_attention/community-kv.patch"
FLASH_ATTENTION_EXTENSION = "community_kv.attention._C"

cuda_home_override = os.environ.get("COMMUNITY_KV_CUDA_HOME")
if cuda_home_override:
    os.environ.setdefault("CUDA_HOME", cuda_home_override)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")

METADATA_COMMANDS = {"dist_info", "egg_info"}
metadata_only = bool(METADATA_COMMANDS.intersection(sys.argv))


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _flash_attention_source() -> Path:
    patch_digest = hashlib.sha256(FLASH_ATTENTION_PATCH.read_bytes()).hexdigest()[:16]
    cache_base = Path(
        os.environ.get(
            "COMMUNITY_KV_BUILD_CACHE",
            Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
            / "community_kv",
        )
    )
    source = (
        cache_base / "sources" / f"flash-attention-{FLASH_ATTENTION_COMMIT[:12]}-{patch_digest}"
    )
    complete = source / ".community-kv-source-ready"
    if complete.is_file():
        return source

    if source.exists():
        shutil.rmtree(source)
    source.parent.mkdir(parents=True, exist_ok=True)
    source.mkdir()

    _run(["git", "init"], cwd=source)
    _run(["git", "remote", "add", "origin", FLASH_ATTENTION_REPOSITORY], cwd=source)
    _run(["git", "fetch", "--depth", "1", "origin", FLASH_ATTENTION_COMMIT], cwd=source)
    _run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=source)
    actual_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    if actual_commit != FLASH_ATTENTION_COMMIT:
        raise RuntimeError(
            f"FlashAttention checkout is {actual_commit}, expected {FLASH_ATTENTION_COMMIT}"
        )
    _run(["git", "apply", "--check", str(FLASH_ATTENTION_PATCH)], cwd=source)
    _run(["git", "apply", str(FLASH_ATTENTION_PATCH)], cwd=source)
    _run(
        ["git", "submodule", "update", "--init", "csrc/cutlass"],
        cwd=source,
    )
    complete.write_text(f"{FLASH_ATTENTION_COMMIT}\n{patch_digest}\n")
    return source


ext_modules: list[Extension] = []
cmdclass = {}
if not metadata_only:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    python_package_roots = {
        Path(package_root)
        for package_root in (*site.getsitepackages(), *sys.path)
        if package_root
    }
    python_package_roots.add(Path(torch.__file__).resolve().parent.parent)
    cuda_vendor_includes = [
        str(include)
        for package_root in python_package_roots
        for include in package_root.glob("nvidia/*/include")
    ]
    cuda_vendor_libraries = [
        str(library)
        for package_root in python_package_roots
        for library in package_root.glob("nvidia/*/lib")
    ]

    def _prepend_search_path(
        environment: dict[str, str],
        name: str,
        paths: list[str],
    ) -> None:
        existing = environment.get(name)
        environment[name] = os.pathsep.join(
            [*paths, *([existing] if existing else [])]
        )

    class CommunityKVBuildExtension(BuildExtension):
        def run(self) -> None:
            source = _flash_attention_source()
            flash_attention_build = (
                Path(self.build_temp) / "flash_attention"
            ).resolve()
            flash_attention_lib = flash_attention_build / "lib"
            flash_attention_temp = flash_attention_build / "temp"
            environment = os.environ.copy()
            environment["FLASH_ATTENTION_FORCE_BUILD"] = "TRUE"
            for feature in (
                "BACKWARD",
                "SPLIT",
                "PAGEDKV",
                "APPENDKV",
                "LOCAL",
                "SOFTCAP",
                "FP16",
                "FP8",
                "VARLEN",
                "CLUSTER",
                "HDIM64",
                "HDIM96",
                "HDIM192",
                "HDIM256",
                "SM80",
            ):
                environment.setdefault(
                    f"FLASH_ATTENTION_DISABLE_{feature}",
                    "TRUE",
                )
            environment.setdefault("MAX_JOBS", "8")
            environment.setdefault("NVCC_THREADS", "4")
            _prepend_search_path(
                environment,
                "CPATH",
                cuda_vendor_includes,
            )
            _prepend_search_path(
                environment,
                "LIBRARY_PATH",
                cuda_vendor_libraries,
            )
            _prepend_search_path(
                environment,
                "LD_LIBRARY_PATH",
                cuda_vendor_libraries,
            )
            _run(
                [
                    sys.executable,
                    "setup.py",
                    "build_ext",
                    f"--build-lib={flash_attention_lib}",
                    f"--build-temp={flash_attention_temp}",
                ],
                cwd=source / "hopper",
                env=environment,
            )

            built_extensions = list((flash_attention_lib / "flash_attn_3").glob("_C*.so"))
            if len(built_extensions) != 1:
                raise RuntimeError(
                    "Expected one built FlashAttention extension, found "
                    f"{len(built_extensions)} under {flash_attention_lib}"
                )
            self._community_kv_flash_attention_extension = built_extensions[0]
            super().run()

        def build_extension(self, extension: Extension) -> None:
            if extension.name == FLASH_ATTENTION_EXTENSION:
                source = self._community_kv_flash_attention_extension
                destination = Path(self.get_ext_fullpath(extension.name))
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                return
            super().build_extension(extension)

    ext_modules = [
        Extension(name=FLASH_ATTENTION_EXTENSION, sources=[]),
        CUDAExtension(
            name="community_kv.attention.kernels._selection_native",
            sources=[
                str(ROOT / "community_kv/attention/kernels/_csrc/bindings.cpp"),
                str(
                    ROOT
                    / "community_kv/attention/kernels/_csrc/descriptor_selection.cu"
                ),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--use_fast_math",
                    "--threads=4",
                    "-lineinfo",
                ],
            },
        ),
        CUDAExtension(
            name="community_kv.graph.leiden._community_kv_leiden",
            include_dirs=cuda_vendor_includes,
            sources=[
                str(ROOT / "community_kv/graph/leiden/_csrc/bindings.cpp"),
                str(ROOT / "community_kv/graph/leiden/_csrc/aggregate.cu"),
                str(ROOT / "community_kv/graph/leiden/_csrc/csr.cu"),
                str(ROOT / "community_kv/graph/leiden/_csrc/helpers.cu"),
                str(ROOT / "community_kv/graph/leiden/_csrc/leiden.cu"),
                str(ROOT / "community_kv/graph/leiden/_csrc/local_moving.cu"),
                str(ROOT / "community_kv/graph/leiden/_csrc/refinement.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--use_fast_math",
                    "--threads=4",
                    "-lineinfo",
                ],
            },
        ),
    ]
    cmdclass = {"build_ext": CommunityKVBuildExtension.with_options(use_ninja=True)}


setup(
    packages=find_packages(
        where=".",
        include=("community_kv", "community_kv.*", "evals", "evals.*"),
    ),
    include_package_data=False,
    package_data={
        "evals": ["resolutions.json"],
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
