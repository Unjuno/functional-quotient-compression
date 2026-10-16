#!/usr/bin/env python3
"""Read-only local preflight. Never downloads, installs, or reads credentials."""
from __future__ import annotations
import argparse, importlib.metadata, json, os, platform, subprocess, sys
from pathlib import Path


def command(*args: str) -> str | None:
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=10, check=False)
        return p.stdout.strip() if p.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def inspect() -> dict:
    packages = {}
    for name in ('torch', 'numpy', 'regex', 'scikit-learn', 'scipy', 'threadpoolctl', 'pytest', 'transformers', 'tokenizers'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    memory = command('sysctl', '-n', 'hw.memsize') if sys.platform == 'darwin' else None
    if memory is None and hasattr(os, 'sysconf'):
        try:
            memory = str(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'))
        except (OSError, ValueError):
            pass
    info = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
            'machine': platform.machine(), 'cpu_count': os.cpu_count(),
            'chip': command('sysctl', '-n', 'machdep.cpu.brand_string') if sys.platform == 'darwin' else platform.processor(),
            'physical_memory_bytes': int(memory) if memory and memory.isdigit() else None,
            'packages': packages, 'git_head': command('git', 'rev-parse', 'HEAD'),
            'git_status': command('git', 'status', '--porcelain'),
            'mps_environment': {k: os.environ.get(k) for k in ('PYTORCH_ENABLE_MPS_FALLBACK', 'PYTORCH_MPS_FAST_MATH', 'PYTORCH_MPS_HIGH_WATERMARK_RATIO')},
            'scope': 'availability only; no inference parity, speed or memory sufficiency claim'}
    if packages['torch']:
        import torch
        info['torch'] = {'mps_built': torch.backends.mps.is_built(), 'mps_available': torch.backends.mps.is_available(),
                         'cuda_available': torch.cuda.is_available(), 'cuda_version': torch.version.cuda}
    return info


def main() -> None:
    if sys.version_info < (3, 11):
        raise SystemExit("The T282 local lane requires Python 3.11 or newer.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError('Use a new report path; existing evidence is never overwritten.')
    data = json.dumps(inspect(), ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(data)
    print(data)


if __name__ == '__main__':
    main()
