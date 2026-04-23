"""
generate_perturbations.py
=========================

Generates text-perturbed variants of Time-MMD preprocessed data for three
multimodal TS models: MM-TSFlib, TaTS, and Aurora (reuses TaTS CSVs).

This script implements conditions C1-C5, C7, C8 at the CSV level.
C6 (unimodal) is NOT a CSV perturbation: it's triggered by CLI flags
(--prompt_weight 0 for MM-TSFlib, --text_emb 0 --prior_weight 0 for TaTS,
--no_text patch for Aurora). This keeps C6 reproducing paper baselines
exactly, per design decision.

WHAT EACH CONDITION TESTS
-------------------------
C1 Original : Unmodified data. Baseline for paper reproducibility.
C2 Empty    : Text = "". Numeric text-derived features also zeroed.
              If C2 ≈ C1, the multimodal model isn't using text-derived signal.
C3 Shuffled : Text column permuted within-domain. Numeric priors left intact.
              If C3 ≈ C1, the text ENCODER isn't contributing semantic signal.
              (Weaker claim than C2 because numeric priors still help.)
C4 CrossDom : Text replaced with paired domain's text (cycled+shuffled).
              If C4 ≈ C3, text isn't domain-specific for the model.
              If C4 << C3, text has domain-specific value.
C5 Constant : Text = "Time series data point." Numeric priors also zeroed.
              Similar to C2 but with non-empty tokens; tests whether the
              PRESENCE of any text affects gating.
C7 Null     : Text = "0". Numeric priors also zeroed.
              Like C2/C5 but shortest non-empty token.
C8 Oracle   : Text encodes ground-truth future values.
              POSITIVE CONTROL — a model that reads text semantically
              MUST benefit. If it doesn't, the probe is broken.

DESIGN DECISIONS (per user)
---------------------------
* For C2/C5/C7: also zero the text-derived NUMERIC columns
  (prior_history_avg, etc.) so these conditions are "no text-derived info."
* For C3/C4: do NOT zero numeric priors; these test the text-encoder path
  specifically, with the prior still available as a cheap signal.
* For C8: do NOT zero numerics; oracle text alone should already saturate
  performance, and the prior can only help.

OUTPUT LAYOUT
-------------
/home/claude/probe_project/data/
    mmtsflib/
        C1_original/seed2021/<DomainDir>/<file>.csv
        C2_empty/seed2021/<DomainDir>/<file>.csv
        ... for each condition x seed x domain
    tats/
        C1_original/seed2024/<Domain>.csv
        ... for each condition x seed x domain
    manifest.json   — what was generated, when, with what parameters

Every perturbed CSV preserves row count, date columns, OT values, and all
non-text-related columns byte-exact with the original. This is enforced
by the validator (see bottom of file).

USAGE
-----
  python3 generate_perturbations.py --repo mmtsflib  # or 'tats', or 'all'
  python3 generate_perturbations.py --repo all --validate_only  # re-check existing
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# ============================================================================
#  Paths & project constants
# ============================================================================

PROJECT_ROOT = Path('/home/karthik/Semantics_Structure_MMTS')
REPOS_DIR = PROJECT_ROOT / 'repos'
DATA_OUT_DIR = PROJECT_ROOT / 'data'

MMTSFLIB_SRC = REPOS_DIR / 'MM-TSFlib' / 'data'
TATS_SRC = REPOS_DIR / 'TaTS' / 'data'


# ============================================================================
#  Canonical domain registry
#
#  The three repos use inconsistent names:
#    - MM-TSFlib: 'Algriculture' [typo], 'Public_Health', 'SocialGood', per-
#                 domain subfolder with a unique CSV filename (e.g.
#                 US_TradeBalance_Month.csv).
#    - TaTS:     'Agriculture', 'Health', 'SocialGood', single file
#                 <Name>.csv directly under data/.
#    - Aurora:   (reuses TaTS files)
#
#  We use a canonical name for each domain (for pairing logic) and map to
#  each repo's file layout.
# ============================================================================

@dataclass(frozen=True)
class DomainSpec:
    canonical: str
    mmtsflib_dir: str      # subfolder name under MM-TSFlib/data/
    mmtsflib_file: str     # csv filename inside that subfolder
    tats_file: str         # filename under TaTS/data/ (also used by Aurora)
    paired: str            # canonical name of the paired domain for C4

DOMAINS: list[DomainSpec] = [
    DomainSpec('Agriculture', 'Algriculture', 'US_RetailBroilerComposite_Month.csv',
               'Agriculture.csv', 'Security'),
    DomainSpec('Climate',     'Climate',      'US_precipitation_month.csv',
               'Climate.csv',     'Energy'),
    DomainSpec('Economy',     'Economy',      'US_TradeBalance_Month.csv',
               'Economy.csv',     'Health'),
    DomainSpec('Energy',      'Energy',       'US_GasolinePrice_Week.csv',
               'Energy.csv',      'Climate'),
    DomainSpec('Environment', 'Environment',  'NewYork_AQI_Day.csv',
               'Environment.csv', 'Environment'),   # self-paired
    DomainSpec('Health',      'Public_Health','US_FLURATIO_Week.csv',
               'Health.csv',      'Economy'),
    DomainSpec('Security',    'Security',     'US_FEMAGrant_Month.csv',
               'Security.csv',    'Agriculture'),
    DomainSpec('SocialGood',  'SocialGood',   'Unadj_UnemploymentRate_ALL_processed.csv',
               'SocialGood.csv',  'Traffic'),
    DomainSpec('Traffic',     'Traffic',      'US_VMT_Month.csv',
               'Traffic.csv',     'SocialGood'),
]

CANON_TO_SPEC = {d.canonical: d for d in DOMAINS}


# ============================================================================
#  Seed & condition constants
# ============================================================================

# Unified seed set across all three models. Previously we had per-model seeds
# (TaTS: 2024-2026, others: 2021-2023) but since TaTS and Aurora share CSVs and
# Aurora's reference uses 2021, we unify. Each seed controls C3 shuffled and
# C4 cross-domain permutations; C1/C2/C5/C7/C8 content is seed-independent.
SEEDS = {
    'mmtsflib': [2021, 2022, 2023],
    'tats':     [2021, 2022, 2023],
    'aurora':   [2021, 2022, 2023],
}

CONDITIONS = ['C1_original', 'C2_empty', 'C3_shuffled', 'C4_crossdomain',
              'C5_constant', 'C7_null', 'C8_oracle']
# C6 absent by design — triggered via CLI at run-time.

# Conditions whose CSV content depends on the seed (i.e. use an RNG).
# Other conditions produce byte-identical CSVs across seeds — we write them once.
SEEDED_CONDITIONS = {'C3_shuffled', 'C4_crossdomain'}

C5_CONSTANT_STR = 'Time series data point.'
C7_NULL_STR = '0'
C8_ORACLE_HORIZON = 8   # how many future OT values to leak in the oracle text


# ============================================================================
#  Repo-specific column specs
#
#  Under features='S' (univariate), which is what actually runs against the
#  preprocessed data, only a small subset of columns is consumed:
#    - MM-TSFlib:  OT (target), Final_Search_{text_len} (text), prior_history_avg
#                  (fed via prior_y). Other his_avg_*, his_std_* are ignored.
#    - TaTS:       OT (target), fact (text), prior_history_avg (prior_y addend).
#    - Aurora:     OT (target), fact (text). prior_history_avg loaded but unused.
# ============================================================================

@dataclass(frozen=True)
class RepoSpec:
    name: str
    text_cols_primary: list[str]
    text_cols_optional: list[str]
    null_numeric_cols: list[str]     # zeroed under C2, C5, C7

MMTSFLIB = RepoSpec(
    name='mmtsflib',
    text_cols_primary=['Final_Search_4'],                  # paper reference
    text_cols_optional=['Final_Search_2', 'Final_Search_6',
                        'Final_Output'],                   # also present
    null_numeric_cols=['prior_history_avg'],
)

TATS = RepoSpec(
    name='tats',
    text_cols_primary=['fact'],
    text_cols_optional=[],   # 'preds' exists but is unused by TaTS loader
    null_numeric_cols=['prior_history_avg'],
)


# ============================================================================
#  Perturbation functions
#
#  Each function returns a NEW DataFrame (never mutates input). Every
#  non-text, non-null-numeric column is preserved byte-exact — the validator
#  enforces this. This is critical for a fair comparison: if perturbing text
#  also accidentally changes OT values, the whole experiment is invalid.
# ============================================================================

def _set_cols(df: pd.DataFrame, cols: Iterable[str], value) -> None:
    """In-place set columns to a value, skipping columns not present."""
    for c in cols:
        if c in df.columns:
            df[c] = value


def c1_original(df: pd.DataFrame, **_) -> pd.DataFrame:
    """C1: unmodified copy."""
    return df.copy()


def c2_empty(df: pd.DataFrame, text_cols, null_numeric_cols, **_) -> pd.DataFrame:
    """C2: text = '', numeric text-derived cols = 0."""
    out = df.copy()
    _set_cols(out, text_cols, '')
    _set_cols(out, null_numeric_cols, 0.0)
    return out


def c3_shuffled(df: pd.DataFrame, text_cols, seed, **_) -> pd.DataFrame:
    """C3: shuffle text column(s) with a fixed permutation.

    NOTE: numeric priors NOT zeroed (per design — C3 probes the text-encoder
    path specifically). Also, we use ONE permutation for ALL text columns,
    so their cross-column alignment is preserved (e.g., Final_Search_2 and
    Final_Search_4 on the same original row still end up on the same shuffled
    row). This matters if a caller ever runs multiple text_cols.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(out))
    for c in text_cols:
        if c in out.columns:
            out[c] = out[c].to_numpy()[perm]
    return out


def c4_crossdomain(df: pd.DataFrame, paired_df: pd.DataFrame, text_cols,
                   seed, **_) -> pd.DataFrame:
    """C4: replace text with paired domain's text, cycled to length, then shuffled.

    The 'cycle' step is deterministic: if paired domain has fewer rows than
    target, we tile (repeat) to reach target length; if it has more, we
    truncate. No randomness here — this step just equalizes lengths.

    The shuffle AFTER cycling is the seed-dependent step. Without shuffling,
    tiled positions would be correlated with the target's row index (e.g. if
    paired domain has exactly half our rows, row N and row N+half would have
    identical tiled text), which could create spurious artifacts.

    For Environment (self-paired), behavior equals C3.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    n = len(out)
    for c in text_cols:
        if c not in out.columns or c not in paired_df.columns:
            continue
        src = paired_df[c].to_numpy()
        # Tile and truncate: repeat src until we have at least n items.
        reps = (n + len(src) - 1) // len(src)
        tiled = np.tile(src, reps)[:n]
        # Shuffle so that there's no positional bias from tiling.
        perm = rng.permutation(n)
        out[c] = tiled[perm]
    return out


def c5_constant(df: pd.DataFrame, text_cols, null_numeric_cols,
                constant=C5_CONSTANT_STR, **_) -> pd.DataFrame:
    """C5: all text = constant string. Numeric priors zeroed."""
    out = df.copy()
    _set_cols(out, text_cols, constant)
    _set_cols(out, null_numeric_cols, 0.0)
    return out


def c7_null(df: pd.DataFrame, text_cols, null_numeric_cols,
            null=C7_NULL_STR, **_) -> pd.DataFrame:
    """C7: text = '0' (minimal non-empty). Numeric priors zeroed."""
    out = df.copy()
    _set_cols(out, text_cols, null)
    _set_cols(out, null_numeric_cols, 0.0)
    return out


def c8_oracle(df: pd.DataFrame, text_cols, target='OT',
              horizon=C8_ORACLE_HORIZON, **_) -> pd.DataFrame:
    """C8: inject ground-truth future values into text.

    For row i, text contains OT[i+1 : i+1+horizon] as numeric literals inside
    the same 'Available facts are as follows:' skeleton that the real text
    uses. This matches the input distribution the model was trained on in
    structure, but its content is a direct leak of the target.

    PEDAGOGICAL NOTE: This is a POSITIVE CONTROL. It answers the question
    "can this model even read text-encoded numbers?" If C8 doesn't beat C1,
    the model is functionally text-blind and our other conditions (C2-C7)
    can't be interpreted as 'the model chose to ignore text' — because the
    model literally can't use text-encoded information even when it's
    maximally useful. Positive controls of this kind are standard in
    mechanistic interpretability (e.g., counterfactual probing).

    BOUNDARY: Rows near the end have fewer than `horizon` future values;
    we just include what's available (maybe zero values at the very last row).
    """
    out = df.copy()
    if target not in out.columns:
        raise ValueError(f'C8 oracle requires target col "{target}" in frame')
    y = out[target].to_numpy()
    n = len(out)
    texts = []
    for i in range(n):
        fut = y[i + 1 : i + 1 + horizon]
        parts = ['Available facts are as follows:']
        for j, v in enumerate(fut):
            parts.append(f'Step+{j+1}: The OT will be {v:.6f}.')
        texts.append(' '.join(parts) + ';')
    for c in text_cols:
        if c in out.columns:
            out[c] = texts
    return out


# Dispatch table: condition name -> (function, kwargs_from_context)
PERTURB_FNS = {
    'C1_original':   c1_original,
    'C2_empty':      c2_empty,
    'C3_shuffled':   c3_shuffled,
    'C4_crossdomain': c4_crossdomain,
    'C5_constant':   c5_constant,
    'C7_null':       c7_null,
    'C8_oracle':     c8_oracle,
}


# ============================================================================
#  Validator: every output must match original on protected columns
# ============================================================================

def validate_perturbation(original: pd.DataFrame, perturbed: pd.DataFrame,
                          allowed_text_cols: list[str],
                          allowed_null_numeric_cols: list[str],
                          condition: str,
                          out_path: Path | None = None) -> list[str]:
    """Return a list of violation messages; empty list = clean.

    If out_path is given, also do a round-trip check: re-read from disk
    with the patched-reader settings (keep_default_na=False) and confirm
    the text columns have expected content — crucial for C2_empty where
    the default pandas reader silently converts '' to NaN.
    """
    errors = []

    # Row count must match.
    if len(original) != len(perturbed):
        errors.append(f'row count changed: {len(original)} -> {len(perturbed)}')
        return errors

    # Columns must match.
    if list(original.columns) != list(perturbed.columns):
        errors.append(f'column layout changed')

    # For C3/C4/C8: numeric priors should NOT have been zeroed.
    # For C2/C5/C7: they SHOULD be zero (if present).
    should_be_zero = condition in ('C2_empty', 'C5_constant', 'C7_null')

    for col in original.columns:
        orig_vals = original[col]
        new_vals = perturbed[col]
        if col in allowed_text_cols:
            continue  # text cols are allowed to change
        if col in allowed_null_numeric_cols:
            if should_be_zero:
                if not (new_vals.fillna(0) == 0).all():
                    errors.append(f'{col}: expected all zero under {condition}')
            else:
                if not orig_vals.equals(new_vals):
                    errors.append(f'{col}: changed under {condition} but should NOT have')
            continue
        # Everything else must be preserved byte-exact.
        if not orig_vals.equals(new_vals):
            # Give a helpful diff.
            diff_mask = (orig_vals != new_vals) & ~(orig_vals.isna() & new_vals.isna())
            errors.append(f'{col}: {diff_mask.sum()} values changed (should be 0)')

    # Round-trip check: read back from disk with keep_default_na=False (the
    # setting a patched downstream loader will use) and verify text cols
    # are correctly preserved. This catches the '' -> NaN CSV bug.
    if out_path is not None and condition == 'C2_empty':
        rt = pd.read_csv(out_path, keep_default_na=False)
        for c in allowed_text_cols:
            if c in rt.columns:
                # Expect every row to be the literal empty string ''.
                non_empty = rt[c].astype(str).str.len() > 0
                if non_empty.any():
                    n_bad = non_empty.sum()
                    errors.append(f'C2 round-trip: {c} has {n_bad} non-empty values '
                                  f'after re-read (expected all "")')

    return errors


# ============================================================================
#  Main driver
# ============================================================================

@dataclass
class ManifestEntry:
    repo: str
    condition: str
    seed: int
    domain: str
    src_file: str
    out_file: str
    n_rows: int
    text_cols_perturbed: list[str]
    numeric_cols_zeroed: list[str]
    n_violations: int
    violations: list[str] = field(default_factory=list)


def _load_mmtsflib_csv(spec: DomainSpec) -> tuple[pd.DataFrame, Path]:
    src = MMTSFLIB_SRC / spec.mmtsflib_dir / spec.mmtsflib_file
    return pd.read_csv(src), src


def _load_tats_csv(spec: DomainSpec) -> tuple[pd.DataFrame, Path]:
    src = TATS_SRC / spec.tats_file
    return pd.read_csv(src), src


def generate_for_repo(repo_spec: RepoSpec,
                      seeds: list[int],
                      out_root: Path,
                      loader_fn) -> list[ManifestEntry]:
    """Generate all condition x seed x domain CSVs for one repo."""
    entries: list[ManifestEntry] = []

    # Preload all domains so C4 can cross-reference.
    loaded: dict[str, pd.DataFrame] = {}
    src_paths: dict[str, Path] = {}
    for spec in DOMAINS:
        try:
            df, src = loader_fn(spec)
            loaded[spec.canonical] = df
            src_paths[spec.canonical] = src
        except FileNotFoundError as e:
            print(f'  [skip] {spec.canonical}: source file not found ({e})')
            continue

    text_cols = repo_spec.text_cols_primary   # primary only (others in appendix)
    numeric_cols = repo_spec.null_numeric_cols

    for condition in CONDITIONS:
        # Non-seeded conditions (C1/C2/C5/C7/C8) produce byte-identical CSVs
        # regardless of seed, so we only write them once. We use a sentinel
        # seed=0 for their on-disk path; the runner substitutes this path
        # regardless of the RunSpec's model-RNG seed.
        is_seeded = condition in SEEDED_CONDITIONS
        condition_seeds = seeds if is_seeded else [0]

        for seed in condition_seeds:
            for spec in DOMAINS:
                if spec.canonical not in loaded:
                    continue
                df = loaded[spec.canonical]

                # Actually apply the perturbation.
                fn = PERTURB_FNS[condition]
                kwargs = dict(text_cols=text_cols,
                              null_numeric_cols=numeric_cols,
                              seed=seed)
                if condition == 'C4_crossdomain':
                    pair_name = spec.paired
                    if pair_name not in loaded:
                        print(f'  [warn] {spec.canonical} paired with {pair_name} '
                              f'but paired not loaded; falling back to self-shuffle')
                        pair_df = df
                    else:
                        pair_df = loaded[pair_name]
                    kwargs['paired_df'] = pair_df
                perturbed = fn(df, **kwargs)

                # Compute what columns changed (for validator + manifest).
                changed_text = [c for c in text_cols if c in df.columns]
                zeroed_num = []
                if condition in ('C2_empty', 'C5_constant', 'C7_null'):
                    zeroed_num = [c for c in numeric_cols if c in df.columns]

                # Write to disk first (validator needs to read it back).
                if repo_spec.name == 'mmtsflib':
                    out_dir = (out_root / 'mmtsflib' / condition /
                               f'seed{seed}' / spec.mmtsflib_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / spec.mmtsflib_file
                else:   # tats / aurora
                    out_dir = out_root / 'tats' / condition / f'seed{seed}'
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / spec.tats_file

                perturbed.to_csv(out_path, index=False,
                                 quoting=csv.QUOTE_NONNUMERIC)

                # Validate (includes round-trip check for C2).
                errors = validate_perturbation(
                    original=df, perturbed=perturbed,
                    allowed_text_cols=changed_text,
                    allowed_null_numeric_cols=numeric_cols,
                    condition=condition,
                    out_path=out_path,
                )

                entries.append(ManifestEntry(
                    repo=repo_spec.name,
                    condition=condition,
                    seed=seed,
                    domain=spec.canonical,
                    src_file=str(src_paths[spec.canonical]),
                    out_file=str(out_path),
                    n_rows=len(perturbed),
                    text_cols_perturbed=changed_text,
                    numeric_cols_zeroed=zeroed_num,
                    n_violations=len(errors),
                    violations=errors,
                ))

    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', choices=['mmtsflib', 'tats', 'all'],
                        default='all')
    parser.add_argument('--out_root', default=str(DATA_OUT_DIR))
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    all_entries: list[ManifestEntry] = []

    if args.repo in ('mmtsflib', 'all'):
        print('=== Generating for MM-TSFlib ===')
        all_entries += generate_for_repo(
            MMTSFLIB, SEEDS['mmtsflib'], out_root, _load_mmtsflib_csv,
        )

    if args.repo in ('tats', 'all'):
        print('=== Generating for TaTS (also serves Aurora) ===')
        all_entries += generate_for_repo(
            TATS, SEEDS['tats'], out_root, _load_tats_csv,
        )

    # Summary.
    total = len(all_entries)
    clean = sum(1 for e in all_entries if e.n_violations == 0)
    print(f'\nSUMMARY: {clean} / {total} files validated clean')
    if clean < total:
        print('VIOLATIONS found — first few:')
        for e in all_entries:
            if e.n_violations > 0:
                print(f'  {e.repo}/{e.condition}/seed{e.seed}/{e.domain}: {e.violations[:2]}')
                break

    manifest = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'project_root': str(PROJECT_ROOT),
        'total_files': total,
        'clean_files': clean,
        'seeds': SEEDS,
        'conditions': CONDITIONS,
        'entries': [asdict(e) for e in all_entries],
    }
    (out_root / 'manifest.json').write_text(json.dumps(manifest, indent=2,
                                                       default=str))
    print(f'manifest written to {out_root}/manifest.json')


if __name__ == '__main__':
    main()