#!/usr/bin/env python3
"""Compile an Inductor GEMM max-autotune path for fake gfx1250 and scan dumps."""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path


def _set_default_env() -> tuple[Path, Path]:
    dump_dir = Path(
        os.environ.setdefault(
            "TRITON_DUMP_DIR", tempfile.mkdtemp(prefix="triton-gfx1250-gemm-")
        )
    )
    cache_dir = Path(
        os.environ.setdefault(
            "TORCHINDUCTOR_CACHE_DIR",
            tempfile.mkdtemp(prefix="inductor-gfx1250-gemm-"),
        )
    )

    defaults = {
        "TORCHINDUCTOR_COMPILE_ONLY_FAKE_ROCM_ARCH": "gfx1250",
        "TORCHINDUCTOR_ENABLE_TDM_CONFIGS": "1",
        "TORCHINDUCTOR_MAX_AUTOTUNE_COMPILE_ONLY": "1",
        "TORCHINDUCTOR_MAX_AUTOTUNE": "1",
        "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS": "ATEN,TRITON",
        "TORCHINDUCTOR_AUTOTUNE_AT_COMPILE_TIME": "0",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "TRITON_ALWAYS_COMPILE": "1",
        "TRITON_KERNEL_DUMP": "1",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)

    return dump_dir, cache_dir


def _avoid_source_tree_import() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cwd = Path.cwd().resolve()
    filtered = []
    for entry in sys.path:
        entry_path = cwd if entry == "" else Path(entry).resolve()
        if entry_path != repo_root:
            filtered.append(entry)
    sys.path[:] = filtered


def _find_matches(root: Path, pattern: re.Pattern[str], limit: int = 20) -> list[str]:
    matches: list[str] = []
    if not root.exists():
        return matches
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        try:
            lines = path.read_text(errors="ignore").splitlines()
        except OSError:
            continue
        for line_no, line in enumerate(lines, 1):
            if pattern.search(line):
                matches.append(f"{path}:{line_no}: {line.strip()[:240]}")
                if len(matches) >= limit:
                    return matches
    return matches


def _print_matches(title: str, matches: list[str]) -> None:
    print(f"\n{title}: {len(matches)} sample(s)")
    for match in matches:
        print(match)


def main() -> int:
    dump_dir, cache_dir = _set_default_env()
    _avoid_source_tree_import()

    import torch
    import torch._inductor.config as inductor_config

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required to trigger Inductor GPU GEMM")

    print(f"python={sys.executable}")
    print(f"torch={torch.__version__}")
    print(f"torch_file={torch.__file__}")
    print(f"torch_hip={torch.version.hip}")
    print(f"fake_arch={inductor_config.compile_only_fake_rocm_arch}")
    print(f"max_autotune={inductor_config.max_autotune}")
    print(f"max_autotune_compile_only={inductor_config.max_autotune_compile_only}")
    print(f"max_autotune_gemm_backends={inductor_config.max_autotune_gemm_backends}")
    print(f"TRITON_DUMP_DIR={dump_dir}")
    print(f"TORCHINDUCTOR_CACHE_DIR={cache_dir}")

    def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a @ b

    torch.manual_seed(0)
    a = torch.randn((256, 256), device="cuda", dtype=torch.float16)
    b = torch.randn((256, 256), device="cuda", dtype=torch.float16)

    compiled = torch.compile(gemm, mode="max-autotune", fullgraph=True)
    out = compiled(a, b)
    torch.cuda.synchronize()
    print(f"compiled_output_shape={tuple(out.shape)} dtype={out.dtype}")

    tdm_pattern = re.compile(
        r"amdg\.(?:async_tdm_[\w_]+|tdm_prefetch)"
        r"|tt\.(?:make_tensor_descriptor|descriptor_load|descriptor_store)"
        r"|llvm\.amdgcn\.tensor\.(?:load\.to\.lds|store\.from\.lds)"
        r"|wait\.tensorcnt",
        re.IGNORECASE,
    )
    target_pattern = re.compile(
        r"gfx1250|DeviceProperties\(type='hip'.*cc='gfx1250'|amdgcn-amd-amdhsa--gfx1250"
    )
    normal_gemm_pattern = re.compile(r"\btt\.dot\b|\btt\.load\b|amdg\.buffer_load")

    tdm_matches = _find_matches(dump_dir, tdm_pattern) + _find_matches(
        cache_dir, tdm_pattern
    )
    target_matches = _find_matches(dump_dir, target_pattern) + _find_matches(
        cache_dir, target_pattern
    )
    normal_matches = _find_matches(dump_dir, normal_gemm_pattern) + _find_matches(
        cache_dir, normal_gemm_pattern
    )

    _print_matches("gfx1250 evidence", target_matches[:20])
    _print_matches("TDM evidence", tdm_matches[:20])
    _print_matches("normal GEMM evidence", normal_matches[:20])
    print(f"\nTDM_FOUND={int(bool(tdm_matches))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
