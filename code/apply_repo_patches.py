"""
apply_repo_patches.py
=====================

Applies surgical, documented patches to each repo so that:

1. `pd.read_csv()` uses `keep_default_na=False` in each loader, which
   makes our C2_empty condition produce literal `''` (empty string)
   instead of pandas-default NaN (which would round-trip to 'nan' via
   f-string coercion in MM-TSFlib or to 'No information available' in
   TaTS/Aurora).

2. TaTS's and Aurora's post-hoc `pd.isnull(text)` replacement block is
   disabled — otherwise it would override our literal `''` values with
   'No information available'.

3. pandas 2.x API fix for `df.drop(cols, 1)` → `df.drop(columns=cols)`.

4. Aurora gets a new `--no_text` CLI flag wired through to set
   `text_input_ids=None` — this is how we do C6 (unimodal) for Aurora,
   per our design decision that C6 uses each paper's reported unimodal
   baseline.

5. TaTS's `exp/exp_basic.py` model_dict is extended from {iTransformer}
   to include 9 backbones available under models/ — their exp_basic
   only registers iTransformer by default, so --backbones Autoformer
   etc. would KeyError without this patch.

Patches are IDEMPOTENT — running twice is safe; the fingerprint check
(looks for the MARKER string after patch) skips already-applied sections.
The fingerprint check runs BEFORE the pattern-match check because some
patches have `new` that CONTAINS `old` (e.g. "insert a line after X"
preserves X), which would otherwise cause double-application.

Each patch writes a `.probe_backup` of the original file on first apply
so we can revert cleanly.

Run with:
    python3 code/apply_repo_patches.py            # applies (idempotent)
    python3 code/apply_repo_patches.py --revert   # restores originals
    python3 code/apply_repo_patches.py --check    # reports marker count

Total edits: 13 (2 MM-TSFlib, 5 TaTS, 6 Aurora).
"""

from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

# Resolve project root dynamically so this works on any box
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPOS = PROJECT_ROOT / 'repos'


# Each patch is a (file_path, list-of-(old_substring, new_substring)) tuple.
# old_substring MUST appear EXACTLY once or the patch is skipped.
# A special marker // PROBE_APPLIED // is embedded into each new_substring
# so we can detect idempotently.

MARKER = '# PROBE_PATCH_APPLIED'


def _mm_tsflib_patches() -> list:
    return [
        # Patch 1: read_csv with keep_default_na=False so '' round-trips
        (REPOS / 'MM-TSFlib' / 'data_provider' / 'data_loader.py',
         [('df_raw = pd.read_csv(os.path.join(self.root_path,\n'
           '                                          self.data_path))',
           'df_raw = pd.read_csv(os.path.join(self.root_path,\n'
           '                                          self.data_path),\n'
           '                             keep_default_na=False)  ' + MARKER),
          # Patch 2: pandas>=2.0 API fix for df.drop (positional axis removed)
          ("data_stamp = df_stamp.drop(['date'], 1).values",
           "data_stamp = df_stamp.drop(columns=['date']).values  " + MARKER),
          ]),
    ]


def _tats_patches() -> list:
    target = REPOS / 'TaTS'
    return [
        (target / 'data_provider' / 'data_loader.py',
         [
             # Patch 1: read_csv with keep_default_na=False
             ('df_raw = pd.read_csv(os.path.join(self.root_path,\n'
              '                                          self.data_path))',
              'df_raw = pd.read_csv(os.path.join(self.root_path,\n'
              '                                          self.data_path),\n'
              '                             keep_default_na=False)  ' + MARKER),
             # Patch 2: same NaN-replacement-with-sentinel fix
             ("        for i in range(len(self.text)):\n"
              "            if pd.isnull(self.text[i][0]):\n"
              "                self.text[i][0] = 'No information available'",
              "        # " + MARKER + " - NaN replacement disabled; '' is a\n"
              "        # legitimate value under C2_empty and must not be\n"
              "        # replaced with a non-empty sentinel.\n"
              "        for i in range(len(self.text)):\n"
              "            if pd.isnull(self.text[i][0]):\n"
              "                self.text[i][0] = ''"),
             # Patch 3: pandas>=2.0 API fix for df.drop
             ("data_stamp = df_stamp.drop(['date'], 1).values",
              "data_stamp = df_stamp.drop(columns=['date']).values  " + MARKER),
         ]),
        # Patch 4 (NEW): register additional backbones in exp_basic model_dict.
        # TaTS ships 9 model files under models/ but only registers iTransformer
        # in exp_basic.py's self.model_dict. We register the rest so our
        # multi-backbone sweep can actually use them. This is a non-semantic
        # change: each Model class is already fully implemented; we just wire
        # them into the dispatcher.
        (target / 'exp' / 'exp_basic.py',
         [
             ("from models import iTransformer",
              "from models import (iTransformer, Autoformer, DLinear, FEDformer,\n"
              "                    FiLM, Informer, PatchTST, Transformer, Crossformer)  " + MARKER),
             ("        self.model_dict = {\n"
              "            'iTransformer': iTransformer\n"
              "        }",
              "        self.model_dict = {  " + MARKER + "\n"
              "            'iTransformer': iTransformer,\n"
              "            'Autoformer': Autoformer,\n"
              "            'DLinear': DLinear,\n"
              "            'FEDformer': FEDformer,\n"
              "            'FiLM': FiLM,\n"
              "            'Informer': Informer,\n"
              "            'PatchTST': PatchTST,\n"
              "            'Transformer': Transformer,\n"
              "            'Crossformer': Crossformer,\n"
              "        }"),
         ]),
    ]


def _aurora_patches() -> list:
    target = REPOS / 'Aurora' / 'TimeMMD'
    return [
        (target / 'data_provider' / 'data_loader.py',
         [
             # Patch 1: read_csv
             ('df_raw = pd.read_csv(os.path.join(self.root_path,\n'
              '                                          self.data_path))',
              'df_raw = pd.read_csv(os.path.join(self.root_path,\n'
              '                                          self.data_path),\n'
              '                             keep_default_na=False)  ' + MARKER),
             # Patch 2: same NaN-replacement disable
             ("        for i in range(len(self.text)):\n"
              "            if pd.isnull(self.text[i][0]):\n"
              "                self.text[i][0] = 'No information available'",
              "        # " + MARKER + " - treat '' as a legitimate C2 value.\n"
              "        for i in range(len(self.text)):\n"
              "            if pd.isnull(self.text[i][0]):\n"
              "                self.text[i][0] = ''"),
             # Patch 3: pandas>=2.0 API fix
             ("data_stamp = df_stamp.drop(['date'], 1).values",
              "data_stamp = df_stamp.drop(columns=['date']).values  " + MARKER),
         ]),
        # Patch 3: add --no_text CLI flag and wire through exp_main to
        # invoke model with text_input_ids=None. This is our C6 path for
        # Aurora per design (the model's modeling_aurora.py already has the
        # `if text_input_ids is not None: ... else: text_features=None` branch;
        # we just need to trigger it).
        (target / 'run_longExp.py',
         [
             # Add the CLI arg right after the random_seed line.
             ("parser.add_argument('--random_seed', type=int, default=2021, help='random seed')",
              "parser.add_argument('--random_seed', type=int, default=2021, help='random seed')\n"
              "parser.add_argument('--no_text', action='store_true',\n"
              "                    help='" + MARKER + " - C6 unimodal: pass None for text_input_ids')"),
         ]),
        (target / 'exp' / 'exp_main.py',
         [
             # Wrap the generate call to pass None when --no_text is on.
             ("pred_y = self.model.generate(inputs=batch_x, text_input_ids=batch_input_ids,",
              "pred_y = self.model.generate(inputs=batch_x,\n"
              "                                             text_input_ids=(None if getattr(self.args, 'no_text', False) else batch_input_ids),  " + MARKER),
             # Same for the other forward path (training/eval).
             ("outputs = self.model(input_ids=batch_x, text_input_ids=batch_input_ids,",
              "outputs = self.model(input_ids=batch_x,\n"
              "                                        text_input_ids=(None if getattr(self.args, 'no_text', False) else batch_input_ids),  " + MARKER),
         ]),
    ]


def apply(path: Path, edits: list[tuple[str, str]]) -> tuple[str, int]:
    """Apply edits to file. Returns (status_message, num_applied).
    Idempotency: `old` string must exist in file to apply. If absent, we
    check whether `new` string fingerprint exists; if yes, treat as already
    applied. If neither, report error."""
    if not path.exists():
        return (f'  [miss] {path}: file does not exist', 0)

    content = path.read_text()
    backup = path.with_suffix(path.suffix + '.probe_backup')

    applied = 0
    skipped = 0
    errors = []
    for (old, new) in edits:
        # Check BEFORE applying: is this patch already applied? Some patches
        # have `new` that CONTAINS `old` (e.g. "insert a line after X"
        # preserves X). In such cases `old in content` is True even after
        # application. So always check the fingerprint first.
        if MARKER in new:
            mpos = new.find(MARKER)
            fp = new[mpos:mpos + len(MARKER) + 50]
        else:
            fp = new[:80]

        if fp in content:
            skipped += 1
        elif old in content:
            content = content.replace(old, new, 1)   # FIRST occurrence only
            applied += 1
        else:
            errors.append(f'pattern not found and no fingerprint match: '
                          f'{old[:60]!r}')

    if errors:
        return (f'  [error] {path}: ' + '; '.join(errors), applied)
    if applied == 0 and skipped > 0:
        return (f'  [skip] {path}: all {skipped} patches already applied', 0)
    if applied == 0:
        return (f'  [noop] {path}: no changes', 0)

    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(content)
    return (f'  [ok]   {path}: applied {applied}, skipped {skipped}', applied)


def revert(path: Path) -> str:
    backup = path.with_suffix(path.suffix + '.probe_backup')
    if not backup.exists():
        return f'  [miss] {path}: no backup'
    shutil.copy2(backup, path)
    backup.unlink()
    return f'  [ok]   {path}: reverted from backup'


def check(path: Path) -> str:
    if not path.exists():
        return f'  [miss] {path}'
    content = path.read_text()
    count = content.count(MARKER)
    return f'  [{count} marker(s)] {path}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--revert', action='store_true')
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()

    all_patches = _mm_tsflib_patches() + _tats_patches() + _aurora_patches()

    if args.check:
        print('=== Patch status ===')
        for path, _ in all_patches:
            print(check(path))
        return

    if args.revert:
        print('=== Reverting patches ===')
        for path, _ in all_patches:
            print(revert(path))
        return

    print('=== Applying patches ===')
    total_applied = 0
    for path, edits in all_patches:
        msg, n = apply(path, edits)
        print(msg)
        total_applied += n
    print(f'\nTotal edits applied: {total_applied}')


if __name__ == '__main__':
    main()