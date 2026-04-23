"""
runners/common.py
=================

Shared infrastructure used by all three model runners.

DESIGN
------
A single run is fully described by a `RunSpec`. Every runner takes a
RunSpec and returns a `RunResult`. Everything else (data staging, CLI
construction, metric parsing, result logging) is wrapped around those
two types.

Why this matters: when we later aggregate 378+ runs into tables and
plots, they're all the same shape — makes paired bootstrap / stats
trivial instead of bespoke-per-model.

TEACHING NOTE
-------------
This design follows the "strong typing, weak dispatch" pattern common
in ML research infra. All the variability lives in the RunSpec
(what we're running), not in the runner's call signature. This makes
it easy to add new conditions, seeds, or even new models later without
changing the orchestrator.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# =============================================================================
#  Data types
# =============================================================================

@dataclass
class RunSpec:
    """Fully-specified single experiment cell.

    One RunSpec -> one runner invocation -> one RunResult JSON file.
    """
    model: str              # 'mmtsflib' | 'tats' | 'aurora'
    condition: str          # 'C1_original' | 'C2_empty' | ... | 'C6_unimodal' | ...
    seed: int               # seed for the model's RNG
    domain: str             # canonical domain name: Agriculture, Climate, ...
    pred_len: int           # forecast horizon
    # Model-specific knobs — None means "use the model's paper-default".
    seq_len: Optional[int] = None
    label_len: Optional[int] = None
    # Backbone: for MM-TSFlib/TaTS we pick one from their model registry;
    # Aurora has no backbone axis (the pretrained architecture is fixed).
    # None means "use the runner's DEFAULT_BACKBONE".
    backbone: Optional[str] = None
    extra_args: dict = field(default_factory=dict)

    def cell_id(self) -> str:
        """Stable, human-readable identifier used for result filenames."""
        bb = f'_{self.backbone}' if self.backbone else ''
        return (f'{self.model}{bb}__{self.condition}__seed{self.seed}__'
                f'{self.domain}__h{self.pred_len}')


@dataclass
class RunResult:
    """Everything we record about a single run."""
    spec: RunSpec
    success: bool
    # Metrics (None if parse failed)
    mse: Optional[float] = None
    mae: Optional[float] = None
    smape: Optional[float] = None          # optional if parseable
    # Provenance
    started_at_utc: str = ''
    wall_time_seconds: float = 0.0
    cli_args: list[str] = field(default_factory=list)
    working_dir: str = ''
    stdout_tail: str = ''                  # last N lines of model stdout
    stderr_tail: str = ''
    error: str = ''                        # error message on failure
    # Crash-resilience / provenance fields
    hostname: str = ''
    probe_repo_git_sha: str = ''
    model_repo_git_sha: str = ''
    python_version: str = ''
    torch_version: str = ''
    cuda_visible_devices: str = ''
    stdout_log_path: str = ''              # full log on disk (not just tail)
    # Optional extras the runner may fill in
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to JSON (dataclasses nested)."""
        return json.dumps(_asdict_nested(self), indent=2, default=str)


def _asdict_nested(obj):
    """asdict() that handles nested dataclasses + datetime / Path."""
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _asdict_nested(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_asdict_nested(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _asdict_nested(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    return obj


# =============================================================================
#  Paths
# =============================================================================

PROJECT_ROOT = Path('/home/karthik/Semantics_Structure_MMTS')
REPOS = PROJECT_ROOT / 'repos'
DATA_ROOT = PROJECT_ROOT / 'data'
RESULTS_ROOT = PROJECT_ROOT / 'results'


def result_path(spec: RunSpec) -> Path:
    """Where a RunResult JSON gets saved for this spec."""
    bb = spec.backbone or 'default'
    return (RESULTS_ROOT / spec.model / bb / spec.condition
            / f'seed{spec.seed}' / f'{spec.domain}_h{spec.pred_len}.json')


# =============================================================================
#  Per-domain data-file mapping
#
#  Necessary because MM-TSFlib uses domain subdirs with varying CSV filenames,
#  while TaTS/Aurora use flat {Domain}.csv. Must stay in sync with
#  DOMAINS in generate_perturbations.py.
# =============================================================================

DOMAIN_FILE_MAP = {
    # canonical -> { 'mmtsflib': (subdir, filename), 'tats': filename }
    'Agriculture':  {'mmtsflib': ('Algriculture', 'US_RetailBroilerComposite_Month.csv'),
                     'tats': 'Agriculture.csv'},
    'Climate':      {'mmtsflib': ('Climate', 'US_precipitation_month.csv'),
                     'tats': 'Climate.csv'},
    'Economy':      {'mmtsflib': ('Economy', 'US_TradeBalance_Month.csv'),
                     'tats': 'Economy.csv'},
    'Energy':       {'mmtsflib': ('Energy', 'US_GasolinePrice_Week.csv'),
                     'tats': 'Energy.csv'},
    'Environment':  {'mmtsflib': ('Environment', 'NewYork_AQI_Day.csv'),
                     'tats': 'Environment.csv'},
    'Health':       {'mmtsflib': ('Public_Health', 'US_FLURATIO_Week.csv'),
                     'tats': 'Health.csv'},
    'Security':     {'mmtsflib': ('Security', 'US_FEMAGrant_Month.csv'),
                     'tats': 'Security.csv'},
    'SocialGood':   {'mmtsflib': ('SocialGood', 'Unadj_UnemploymentRate_ALL_processed.csv'),
                     'tats': 'SocialGood.csv'},
    'Traffic':      {'mmtsflib': ('Traffic', 'US_VMT_Month.csv'),
                     'tats': 'Traffic.csv'},
}


# Conditions whose CSV contents are the same regardless of seed. These are
# stored on disk under a sentinel seed=0 path. The runner's RunSpec.seed
# still controls the MODEL's RNG (weight init, dropout, etc); only the CSV
# location is deduplicated.
# (Only C3 is seed-dependent after we switched C4 to date-aligned.)
_SEEDED_DATA_CONDITIONS = {'C3_shuffled'}


def resolve_data_path(spec: RunSpec) -> tuple[Path, str]:
    """Return (root_path, data_filename) for this spec.

    The runner passes these to the model's CLI as --root_path and --data_path.
    This function implements the mapping: which of our perturbation CSVs
    corresponds to this cell?

    SPECIAL CASE 1: C6_unimodal has no pre-generated CSV — it's a CLI-flag
    condition that uses C1_original data plus an unimodal flag.

    SPECIAL CASE 2: Non-seeded conditions (C1/C2/C5/C7/C8) are stored under
    seed0/ since their content doesn't depend on the seed. We look there
    regardless of the RunSpec's seed.
    """
    condition = 'C1_original' if spec.condition == 'C6_unimodal' else spec.condition
    # Data-location seed: only C3/C4 actually vary with seed
    data_seed = spec.seed if condition in _SEEDED_DATA_CONDITIONS else 0

    if spec.model == 'mmtsflib':
        subdir, filename = DOMAIN_FILE_MAP[spec.domain]['mmtsflib']
        root = DATA_ROOT / 'mmtsflib' / condition / f'seed{data_seed}' / subdir
        return root, filename
    elif spec.model in ('tats', 'aurora'):
        filename = DOMAIN_FILE_MAP[spec.domain]['tats']
        root = DATA_ROOT / 'tats' / condition / f'seed{data_seed}'
        return root, filename
    raise ValueError(f'unknown model: {spec.model}')


# =============================================================================
#  Metric parsers — shared utilities
#
#  Each model prints metrics differently. The shared parser looks for common
#  patterns across logs. Per-model runners can extend this with model-specific
#  patterns. Keep these regex parsers simple and well-tested.
# =============================================================================

import re

# Patterns seen across these three libraries (all inherit from Time-Series-Library)
_METRIC_PATTERNS = [
    # "mse:0.2345, mae:0.1234"
    re.compile(r'mse\s*[:=]\s*([0-9.eE+-]+).*?mae\s*[:=]\s*([0-9.eE+-]+)', re.IGNORECASE),
    # "MSE: 0.234  MAE: 0.123"
    re.compile(r'MSE\s*[:=]\s*([0-9.eE+-]+).*?MAE\s*[:=]\s*([0-9.eE+-]+)', re.IGNORECASE),
]


def parse_metrics_from_log(log_text: str) -> dict:
    """Extract MSE/MAE (and optionally SMAPE) from a log string.

    Returns {'mse': float|None, 'mae': float|None, 'smape': float|None}.
    Picks the LAST occurrence in the log (test-set metrics usually print
    last). Returns None for any metric not found.
    """
    result = {'mse': None, 'mae': None, 'smape': None}
    # Find the last MSE/MAE pair in the log.
    last_match = None
    for pat in _METRIC_PATTERNS:
        for m in pat.finditer(log_text):
            last_match = m
    if last_match:
        try:
            result['mse'] = float(last_match.group(1))
            result['mae'] = float(last_match.group(2))
        except (ValueError, IndexError):
            pass
    # Optional SMAPE.
    smape_match = re.search(r'smape\s*[:=]\s*([0-9.eE+-]+)', log_text, re.IGNORECASE)
    if smape_match:
        try:
            result['smape'] = float(smape_match.group(1))
        except ValueError:
            pass
    return result


# =============================================================================
#  Subprocess invocation — the actual model launch
# =============================================================================

def run_subprocess(cli_args: list[str], cwd: Path,
                   env_extra: Optional[dict] = None,
                   timeout: Optional[float] = None) -> tuple[int, str, str]:
    """Run a subprocess, capture stdout/stderr, return (returncode, stdout, stderr).

    Lets the subprocess stream to console in real-time if requested via the
    PROBE_STREAM=1 env var — useful for debugging a single cell interactively.
    """
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)

    stream = os.environ.get('PROBE_STREAM', '') == '1'
    if stream:
        # Tee to console AND capture.
        proc = subprocess.Popen(
            cli_args, cwd=str(cwd), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        out_lines = []
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            out_lines.append(line)
        proc.wait(timeout=timeout)
        return proc.returncode, ''.join(out_lines), ''
    else:
        try:
            proc = subprocess.run(
                cli_args, cwd=str(cwd), env=env,
                capture_output=True, text=True, timeout=timeout,
            )
            return proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as e:
            return -1, e.stdout or '', (e.stderr or '') + '\n[TIMEOUT]'


def tail(text: str, n_lines: int = 40) -> str:
    """Return the last n_lines of text."""
    lines = text.splitlines()
    return '\n'.join(lines[-n_lines:])


# =============================================================================
#  Result persistence & resume — crash-resilient
# =============================================================================

LOGS_ROOT = PROJECT_ROOT / 'logs'     # per-cell full stdout/stderr
SWEEP_LOG = PROJECT_ROOT / 'sweep_log.jsonl'   # append-only global log


def running_marker_path(spec: 'RunSpec') -> Path:
    """Path to a 'this cell is in progress' marker file."""
    return result_path(spec).with_suffix('.running')


def full_log_path(spec: 'RunSpec') -> Path:
    """Path to the full stdout/stderr log for a cell."""
    bb = spec.backbone or 'default'
    return (LOGS_ROOT / spec.model / bb / spec.condition
            / f'seed{spec.seed}' / f'{spec.domain}_h{spec.pred_len}.log')


def _atomic_write(path: Path, content: str) -> None:
    """Write content to path such that readers NEVER see a half-written file.
    Implementation: write to path.tmp, fsync, rename (rename is atomic on
    POSIX). If we crash mid-write, the target path is unchanged."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(path))


def save_result(result: 'RunResult') -> Path:
    """Atomically write RunResult as JSON. Also remove the in-progress
    marker and append to the sweep log."""
    path = result_path(result.spec)
    _atomic_write(path, result.to_json())

    # Clean up the in-progress marker
    marker = running_marker_path(result.spec)
    if marker.exists():
        marker.unlink()

    # Append a one-line summary to the global sweep log
    try:
        LOGS_ROOT.mkdir(parents=True, exist_ok=True)
        with open(SWEEP_LOG, 'a') as f:
            summary = {
                'ts': now_utc(),
                'cell_id': result.spec.cell_id(),
                'success': result.success,
                'mse': result.mse,
                'mae': result.mae,
                'wall_s': round(result.wall_time_seconds, 1),
                'error': (result.error[:150] if result.error else ''),
            }
            f.write(json.dumps(summary) + '\n')
    except OSError:
        pass   # never let logging fail a run

    return path


def mark_running(spec: 'RunSpec') -> None:
    """Write an in-progress marker file. If we crash during the run, this
    marker survives and tells the resume logic that the cell is stale
    and should be re-run."""
    marker = running_marker_path(spec)
    _atomic_write(marker, json.dumps({'started_at_utc': now_utc(),
                                       'pid': os.getpid()}))


def already_done(spec: 'RunSpec') -> bool:
    """True if a successful result already exists for this spec AND no
    in-progress marker is present (marker means previous attempt crashed)."""
    p = result_path(spec)
    marker = running_marker_path(spec)
    if marker.exists():
        return False   # previous attempt crashed; re-run
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text())
        return bool(data.get('success'))
    except (json.JSONDecodeError, OSError):
        return False


def clear_stale_markers() -> int:
    """At sweep startup, clean up any markers from previous crashed runs.
    Returns the count cleared."""
    count = 0
    if not RESULTS_ROOT.exists():
        return 0
    for m in RESULTS_ROOT.rglob('*.running'):
        m.unlink()
        count += 1
    return count


# =============================================================================
#  Provenance helpers
# =============================================================================

def _safe_git_sha(repo_dir: Path) -> str:
    """Return short git SHA of a repo, or 'nogit' if unavailable."""
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', '--short=12', 'HEAD'],
            cwd=str(repo_dir), stderr=subprocess.DEVNULL, text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return 'nogit'


def collect_provenance(model: str) -> dict:
    """Gather git SHAs, python/torch versions, hostname for this run."""
    import platform, socket
    repo_map = {
        'mmtsflib': REPOS / 'MM-TSFlib',
        'tats':     REPOS / 'TaTS',
        'aurora':   REPOS / 'Aurora',
    }
    try:
        import torch
        torch_v = torch.__version__
    except ImportError:
        torch_v = 'unavailable'
    return {
        'hostname': socket.gethostname(),
        'probe_repo_git_sha': _safe_git_sha(PROJECT_ROOT),
        'model_repo_git_sha': _safe_git_sha(repo_map.get(model, PROJECT_ROOT)),
        'python_version': platform.python_version(),
        'torch_version': torch_v,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
    }


# =============================================================================
#  Runner dispatch
#
#  The orchestrator calls run_spec(spec), which dispatches to the right
#  per-model runner. Keeps the orchestrator model-agnostic.
# =============================================================================

def run_spec(spec: RunSpec) -> RunResult:
    """Top-level runner. Dispatches based on spec.model."""
    # Late imports so each module's own imports don't leak globally.
    if spec.model == 'aurora':
        from runners.aurora_runner import run_aurora
        return run_aurora(spec)
    if spec.model == 'tats':
        from runners.tats_runner import run_tats
        return run_tats(spec)
    if spec.model == 'mmtsflib':
        from runners.mmtsflib_runner import run_mmtsflib
        return run_mmtsflib(spec)
    raise ValueError(f'unknown model: {spec.model}')


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()
