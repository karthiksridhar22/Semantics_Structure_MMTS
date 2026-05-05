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

Total edits: 19 (2 MM-TSFlib, 11 TaTS, 6 Aurora).

The TaTS P-fix-detach group (3 edits) wires a CLI flag
`--fix_text_grad` through `run.py` to `Exp_Long_Term_Forecast`, and
gates the two `.detach()` calls in `train()` (lines 519/524 of
`exp/exp_long_term_forecasting.py`) on that flag. Default OFF (the
flag must be passed explicitly) so all pre-existing result JSONs
remain bit-comparable; turning it on lets gradients reach the
text-projection MLP psi (`self.mlp`) so its optimiser
(`model_optim_mlp`) actually updates weights instead of stepping on
zero gradients. See `PAPER_PLAN_V2.md` Appendix F for the audit and
`repos/TaTS/exp/exp_long_term_forecasting.py` lines 519/524.

The TaTS P-fix-inplace-norm group (3 edits, applied unconditionally)
replaces the in-place `x_enc /= stdev` in iTransformer.py, FiLM.py
and PatchTST.py forecast paths with the out-of-place equivalent
`x_enc = x_enc / stdev`. Forward values are bit-identical; only the
autograd graph topology changes. Without this fix, --fix_text_grad
on raises "one of the variables needed for gradient computation has
been modified by an inplace operation" because var(x_enc) saved x_enc
for backward and the /= mutated it. We patch only the long-term
`forecast` function (the only path reachable for our task);
imputation/anomaly_detection/classification are left untouched.
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
        # Patches 5-7 (P-fix-detach): wire a `--fix_text_grad` CLI flag
        # through run.py and Exp_Long_Term_Forecast so the two .detach()
        # calls in train() that sever gradient flow into the text-projection
        # MLP can be toggled off. Upstream TaTS unconditionally detaches the
        # text/numeric concatenation before the backbone, which leaves the
        # MLP at random init even though `model_optim_mlp` is constructed
        # and stepped each iteration. With --fix_text_grad on, gradient
        # reaches `self.mlp.parameters()` and the optimiser updates them.
        # Default OFF preserves bit-equivalence with pre-patch result JSONs.
        # Patch 5: declare --fix_text_grad in run.py.
        (target / 'run.py',
         [
             ("    parser.add_argument('--seed', type=int, default=2024, help='random seed')",
              "    parser.add_argument('--seed', type=int, default=2024, help='random seed')\n"
              "    parser.add_argument('--fix_text_grad', action='store_true',\n"
              "                        help='" + MARKER + " - keep gradient flow through the\\n"
              "                              text-projection MLP psi by skipping the .detach()\\n"
              "                              calls in train() that sever its training signal.')"),
         ]),
        # Patch 6: forward args.fix_text_grad into Exp.__init__ (via configs).
        # configs is just `args` (line 48: `configs=args`), so we set
        # self.fix_text_grad early so the `getattr` fallback in train() picks it up.
        # We bind near `self.use_fullmodel=configs.use_fullmodel` (line 58).
        (target / 'exp' / 'exp_long_term_forecasting.py',
         [
             ("        self.use_fullmodel=configs.use_fullmodel",
              "        self.use_fullmodel=configs.use_fullmodel\n"
              "        # " + MARKER + " - off by default; preserves upstream TaTS behaviour.\n"
              "        self.fix_text_grad = bool(getattr(configs, 'fix_text_grad', False))\n"
              "        if self.fix_text_grad:\n"
              "            print('[TaTS patch] fix_text_grad=ON: gradients into self.mlp (text-projection) "
              "are enabled.')"),
             # Patch 7: gate the two .detach() calls in train(). The block
             # uniquely matches train() (test() has different surrounding
             # context: `prompt_emb = self.mlp(prompt_emb)` instead of
             # `prompt_emb = self.mlp(batch_text_embeddings)`).
             ("                prompt_emb = self.mlp(batch_text_embeddings) \n"
              "\n"
              "                # decoder input\n"
              "                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()\n"
              "                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)\n"
              "\n"
              "                # batch_x is [bsz, seq_len, num_vars], prompt_emb is [bsz, seq_len, text_embedding_dim]. concatenate them in the last dimension\n"
              "                batch_x = torch.cat([batch_x, prompt_emb], dim=-1).detach()\n"
              "\n"
              "                # dec_inp is [bsz, label_len + pred_len, num_vars], where only label_len is the true data, the rest is 0\n"
              "                text_dec_inp = torch.zeros((self.args.batch_size, self.args.pred_len, self.text_embedding_dim)).to(self.device)\n"
              "                text_dec_inp = torch.cat([prompt_emb[:, :self.args.label_len, :], text_dec_inp], dim=1).float().to(self.device)\n"
              "                dec_inp = torch.cat([dec_inp, text_dec_inp], dim=-1).detach()",
              "                prompt_emb = self.mlp(batch_text_embeddings) \n"
              "\n"
              "                # decoder input\n"
              "                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()\n"
              "                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)\n"
              "\n"
              "                # batch_x is [bsz, seq_len, num_vars], prompt_emb is [bsz, seq_len, text_embedding_dim]. concatenate them in the last dimension\n"
              "                # " + MARKER + " - upstream unconditionally detaches here, severing\n"
              "                # gradients into self.mlp. With fix_text_grad on, we keep the graph.\n"
              "                batch_x = torch.cat([batch_x, prompt_emb], dim=-1)\n"
              "                if not getattr(self, 'fix_text_grad', False):\n"
              "                    batch_x = batch_x.detach()\n"
              "\n"
              "                # dec_inp is [bsz, label_len + pred_len, num_vars], where only label_len is the true data, the rest is 0\n"
              "                text_dec_inp = torch.zeros((self.args.batch_size, self.args.pred_len, self.text_embedding_dim)).to(self.device)\n"
              "                text_dec_inp = torch.cat([prompt_emb[:, :self.args.label_len, :], text_dec_inp], dim=1).float().to(self.device)\n"
              "                dec_inp = torch.cat([dec_inp, text_dec_inp], dim=-1)\n"
              "                if not getattr(self, 'fix_text_grad', False):\n"
              "                    dec_inp = dec_inp.detach()"),
         ]),
        # Patches 8-10 (P-fix-inplace-norm): replace `x_enc /= stdev` with the
        # out-of-place `x_enc = x_enc / stdev` in the long-term `forecast`
        # paths of iTransformer.py, FiLM.py and PatchTST.py. Forward values
        # are bit-identical (the in-place divide and the out-of-place divide
        # produce the same numbers); only the autograd graph differs. With
        # --fix_text_grad on, gradients flow from `loss` back through the
        # backbone to `batch_x` and into `self.mlp`. The chain hits
        # `stdev = sqrt(var(x_enc - means))`, which saves `x_enc - means`
        # for backward, and the very next line `x_enc /= stdev` mutates
        # that saved tensor in-place — autograd then raises:
        #   "one of the variables needed for gradient computation has been
        #    modified by an inplace operation: ... output 0 of DivBackward0
        #    is at version 1; expected version 0".
        # We touch ONLY the `forecast` function in each model (the only
        # path reachable when task_name='long_term_forecast'); the other
        # functions (imputation/anomaly_detection/classification) keep the
        # in-place op untouched to minimise diff vs upstream. Applied
        # unconditionally because the change is mathematically identical.
        # Patch 8: iTransformer.forecast — disambiguated by `_, _, N` (the
        # other two iTransformer functions use `_, L, N`).
        (target / 'models' / 'iTransformer.py',
         [
             ("        x_enc /= stdev\n"
              "\n"
              "        _, _, N = x_enc.shape",
              "        # " + MARKER + " (iT-forecast) in-place /= breaks autograd when\n"
              "        # gradients must flow back through batch_x to self.mlp.\n"
              "        x_enc = x_enc / stdev\n"
              "\n"
              "        _, _, N = x_enc.shape"),
         ]),
        # Patch 9: FiLM.forecast — disambiguated by the unique signature
        # `def forecast(self, x_enc, x_mark_enc, x_dec_true, x_mark_dec)`
        # (note `x_dec_true`, which the imputation/anomaly variants lack).
        (target / 'models' / 'FiLM.py',
         [
             ("    def forecast(self, x_enc, x_mark_enc, x_dec_true, x_mark_dec):\n"
              "        # Normalization from Non-stationary Transformer\n"
              "        means = x_enc.mean(1, keepdim=True).detach()\n"
              "        x_enc = x_enc - means\n"
              "        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()\n"
              "        x_enc /= stdev",
              "    def forecast(self, x_enc, x_mark_enc, x_dec_true, x_mark_dec):\n"
              "        # Normalization from Non-stationary Transformer\n"
              "        means = x_enc.mean(1, keepdim=True).detach()\n"
              "        x_enc = x_enc - means\n"
              "        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()\n"
              "        # " + MARKER + " (FiLM-forecast) in-place /= breaks autograd when\n"
              "        # gradients must flow back through batch_x to self.mlp.\n"
              "        x_enc = x_enc / stdev"),
         ]),
        # Patch 10: PatchTST.forecast — disambiguated by including the
        # function signature `def forecast(self, x_enc, x_mark_enc, x_dec,
        # x_mark_dec)`; PatchTST has 4 functions with the same /= body but
        # only `forecast` carries that signature.
        (target / 'models' / 'PatchTST.py',
         [
             ("    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):\n"
              "        # Normalization from Non-stationary Transformer\n"
              "        means = x_enc.mean(1, keepdim=True).detach()\n"
              "        x_enc = x_enc - means\n"
              "        stdev = torch.sqrt(\n"
              "            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)\n"
              "        x_enc /= stdev",
              "    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):\n"
              "        # Normalization from Non-stationary Transformer\n"
              "        means = x_enc.mean(1, keepdim=True).detach()\n"
              "        x_enc = x_enc - means\n"
              "        stdev = torch.sqrt(\n"
              "            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)\n"
              "        # " + MARKER + " (PatchTST-forecast) in-place /= breaks autograd\n"
              "        # when gradients must flow back through batch_x to self.mlp.\n"
              "        x_enc = x_enc / stdev"),
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