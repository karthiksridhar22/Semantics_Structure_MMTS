"""
aurora_probes.py
================

STANDALONE diagnostic probes for Aurora. Completely independent of
the main experiment pipeline — run after your main sweep completes.

Probes:
    A. Gradient norm at the text encoder's input embeddings.
       Measures: does text content drive the gradient? If C1's grad norm
       is indistinguishable from C3's, the model is propagating gradients
       into the text path but the content is being ignored.

    B. Cross-attention entropy at Aurora's TextGuider.
       Measures: how peaked is the attention distribution over text
       tokens? If C1's entropy is identical to C3's, the attention
       pattern isn't differentiated by text content.

    C. Output-prediction divergence between C1 (real text) and C2 (empty).
       Measures: does the forecast change when text content changes?

OUTPUT-INVARIANCE GUARANTEE
---------------------------
This script never writes to the main results/ tree. It writes to
probes/<cell_id>.json. The main sweep is unaffected.

Within this script, probes A and B use PyTorch forward/backward hooks that
are DETACHABLE. We validate output invariance by comparing predictions
with hooks attached vs. without hooks attached — these must match bitwise.

USAGE
-----
    conda activate aurora
    export AURORA_WEIGHTS=$(pwd)/weights/aurora
    python3 code/aurora_probes.py \
        --conditions C1_original C3_shuffled C8_oracle \
        --domains Economy Health Energy \
        --seeds 2021 --pred_lens 8

    # Special validation mode: verify hooks don't change outputs
    python3 code/aurora_probes.py --validate_invariance

WHY AURORA ONLY?
----------------
Aurora has an explicit TextEncoder + cross-attention (TextGuider) that
are clean hook points. MM-TSFlib's fusion is a weighted sum (no
cross-attention), and TaTS precomputes text embeddings at data-loading
time (no live text encoder to hook during training). Aurora is also
where your current results show the smallest multimodal/unimodal gap —
probes here are the most informative for your paper's narrative.
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

PROJECT_ROOT = _HERE.parent
PROBES_ROOT = PROJECT_ROOT / 'probes'
DATA_ROOT = PROJECT_ROOT / 'data'
REPOS = PROJECT_ROOT / 'repos'

# Add Aurora's code path
sys.path.insert(0, str(REPOS / 'Aurora' / 'TimeMMD'))


# =============================================================================
#  Per-domain Aurora defaults (seq_len, inference_token_len, batch_size)
#  Mirror of the reference script scripts/run_aurora_timemmd_zero_shot.sh
# =============================================================================

AURORA_DEFAULTS = {
    'Agriculture':  (192,  48,  256),
    'Climate':      (192,  48,  256),
    'Economy':      (192,  48,  256),
    'Energy':       (1056, 48,  256),
    'Environment':  (528,  48,  256),
    'Health':       (96,   48,  256),
    'Security':     (220,  24,  256),
    'SocialGood':   (192,  48,  256),
    'Traffic':      (96,   48,  256),
}


# =============================================================================
#  Aurora-internal accessors — handle path variations safely
# =============================================================================
#
# AuroraForPrediction wraps an inner AuroraModel as `self.model`. The text
# components (TextEncoder + TextGuider) live on that inner model, NOT on the
# outer AuroraForPrediction. Earlier versions of this file accessed them as
# `model.TextEncoder` directly, which raised AttributeError. These helpers
# walk plausible paths and return the right submodule, or raise a clear
# error listing what we tried — so any future Aurora rewrap is easy to fix.

def _find_text_encoder(aurora_for_prediction):
    """Returns the TextEncoder module."""
    candidates = [
        ('model.TextEncoder', lambda m: m.model.TextEncoder),
        ('TextEncoder',       lambda m: m.TextEncoder),
    ]
    return _try_paths(aurora_for_prediction, candidates, 'TextEncoder')


def _find_text_guider(aurora_for_prediction):
    """Returns the TextGuider (cross-attention) module."""
    candidates = [
        ('model.TextGuider', lambda m: m.model.TextGuider),
        ('TextGuider',       lambda m: m.TextGuider),
    ]
    return _try_paths(aurora_for_prediction, candidates, 'TextGuider')


def _try_paths(obj, candidates, label):
    last = None
    for desc, fn in candidates:
        try:
            return fn(obj)
        except AttributeError as e:
            last = (desc, e)
            continue
    tried = ', '.join(d for d, _ in candidates)
    raise AttributeError(
        f"Could not locate Aurora's {label} via any of: [{tried}]. "
        f"Last error at '{last[0]}': {last[1]}"
    )


# =============================================================================
#  Hook containers — minimal, self-contained
# =============================================================================

class GradNormProbe:
    """Probe A: L2-norm of grad at Aurora's text-encoder OUTPUT
    (the distilled `output_tokens` of shape [B, num_distill, hidden]).

    IMPORTANT — why we hook the encoder *output*, not BERT's input
    embeddings: in Aurora, BERT's parameters are frozen with
    `param.requires_grad = False`. The output tensor of BERT's
    `embeddings.word_embeddings` therefore has `requires_grad = False`
    (no live gradient path exists *into* a frozen-only subgraph from
    here — gradients are only routed through tensors that are upstream
    of trainable parameters). Hooking there yields no gradients.

    Aurora's TextEncoder.forward additionally applies `self.projection`
    (trainable) and `self.cross_text` (trainable Transformer decoder
    that distills BERT's per-token features into `num_distill=10`
    tokens via cross-attention). Hooking the *output* of TextEncoder
    sits right above this trainable stack, so the captured tensor's
    .grad is non-zero whenever loss depends on text — i.e. exactly
    what we want to measure.

    Operationally, .grad ≈ 0 at this hook point means the model treats
    the text-distilled tokens as pass-through noise; .grad >> 0 means
    the model is sensitive to text features.

    Usage:
        probe = GradNormProbe()
        probe.attach(<TextEncoder>)
        ...forward + backward()...
        probe.record()
        probe.detach()
        stats = probe.summary()
    """
    def __init__(self):
        self.handle = None
        self.captured = None   # the tensor whose grad we want
        self.norms = []

    def _fwd(self, module, inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        if torch.is_tensor(tensor) and tensor.requires_grad:
            tensor.retain_grad()
            self.captured = tensor

    def attach(self, module):
        self.handle = module.register_forward_hook(self._fwd)

    def detach(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def record(self):
        """Call AFTER backward(). Reads .grad from the captured tensor."""
        if self.captured is None or self.captured.grad is None:
            return
        g = self.captured.grad.detach()
        # Per-element L2 norm (sqrt of mean of squares)
        norm = float((g.pow(2).mean()).sqrt().item())
        self.norms.append(norm)
        self.captured = None

    def summary(self):
        if not self.norms:
            return {}
        import statistics as s
        return {
            'gradnorm_mean': s.mean(self.norms),
            'gradnorm_std': s.stdev(self.norms) if len(self.norms) > 1 else 0.0,
            'gradnorm_n': len(self.norms),
        }


class AttentionProbe:
    """Probe B: entropy stats of cross-attention over text tokens.

    Aurora's TextGuider is an AuroraAttention whose forward returns
    (output, attn_weights) when output_attentions=True. We capture
    attn_weights shape (B, num_heads, L_q, L_k).
    """
    def __init__(self):
        self.handle = None
        self.stats = []

    def _fwd(self, module, inputs, output):
        try:
            if not (isinstance(output, tuple) and len(output) >= 2):
                return
            attn_scores = output[1]
            if attn_scores is None or not torch.is_tensor(attn_scores):
                return
            # Aurora's AuroraAttention returns the *pre-softmax* scaled
            # dot-product scores as `output[1]`, NOT the softmax weights.
            # We must softmax over the key dimension before any entropy
            # / max-weight computation. Variable name in the upstream code
            # is misleading; this was a real bug in earlier probe runs.
            attn_logits = attn_scores.detach().float()
            attn = torch.softmax(attn_logits, dim=-1)
            L_k = attn.shape[-1]
            if L_k <= 1:
                # Degenerate: 1-key attention has zero entropy and log(1) = 0,
                # so the normalized entropy is undefined. Record a flag.
                self.stats.append({
                    'attn_error': f'degenerate L_k={L_k}',
                    'attn_L_k': int(L_k),
                })
                return
            # Shannon entropy over the key (text-token) dimension; mean
            # across heads, queries, batch. Normalised by log(L_k) so 1.0
            # means uniform (model attends to all tokens equally).
            eps = 1e-12
            H = -(attn * (attn + eps).log()).sum(dim=-1)
            entropy_rel = float((H.mean() / math.log(L_k)).item())
            # Also: max attention weight — how peaked is the peak?
            max_w = float(attn.max(dim=-1).values.mean().item())
            self.stats.append({
                'attn_entropy_rel_uniform': entropy_rel,
                'attn_max_weight': max_w,
                'attn_L_k': int(L_k),
            })
        except Exception as e:
            self.stats.append({'attn_error': f'{type(e).__name__}: {e}'})

    def attach(self, module):
        self.handle = module.register_forward_hook(self._fwd)

    def detach(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def summary(self):
        if not self.stats:
            return {}
        import statistics as s
        clean = [x for x in self.stats if 'attn_error' not in x]
        if not clean:
            return {'attn_errors': [x.get('attn_error') for x in self.stats][:3]}
        return {
            'attn_entropy_rel_mean': s.mean(x['attn_entropy_rel_uniform'] for x in clean),
            'attn_max_weight_mean': s.mean(x['attn_max_weight'] for x in clean),
            'attn_n_batches': len(clean),
        }


# =============================================================================
#  Aurora-specific data path resolution
# =============================================================================

# Non-seeded conditions live under seed0/ on disk. The orchestrator and
# probe both rewrite seed -> 0 for these so all RNG seeds read the same CSV.
NON_SEEDED = {'C1_original', 'C2_empty', 'C5_constant', 'C7_null', 'C8_oracle',
              'C4_crossdomain', 'C6_unimodal', 'C9_zero_priors'}


def resolve_data_path(condition, seed, domain):
    """Where is this cell's CSV? Mirrors logic in runners/common.py."""
    # C6 substitutes C1 data (unimodal is a CLI-only condition)
    data_condition = 'C1_original' if condition == 'C6_unimodal' else condition
    data_seed = 0 if data_condition in NON_SEEDED else seed
    root = DATA_ROOT / 'tats' / data_condition / f'seed{data_seed}'
    return root, f'{domain}.csv'


def probe_path(condition, seed, domain, pred_len):
    return (PROBES_ROOT / 'aurora'
            / condition / f'seed{seed}'
            / f'{domain}_h{pred_len}.json')


# =============================================================================
#  Aurora model + data loading helpers
# =============================================================================

def load_aurora_model_and_dataset(condition, seed, domain, pred_len):
    """Set up model + dataloader for one cell. Returns (model, loader, device)."""
    from aurora.modeling_aurora import AuroraForPrediction
    from data_provider.data_loader import Dataset_TimeMMD
    from torch.utils.data import DataLoader

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weights = os.environ.get('AURORA_WEIGHTS',
                             str(PROJECT_ROOT / 'weights' / 'aurora'))
    if not Path(weights).exists():
        raise FileNotFoundError(
            f'Aurora weights not found at {weights}. Set AURORA_WEIGHTS '
            f'env var or download via huggingface_hub.snapshot_download.'
        )

    model = AuroraForPrediction.from_pretrained(weights).to(device)
    model.eval()

    root_path, data_file = resolve_data_path(condition, seed, domain)
    if not (root_path / data_file).exists():
        raise FileNotFoundError(f'CSV not found: {root_path / data_file}')

    seq_len, inference_token_len, batch_size = AURORA_DEFAULTS[domain]

    args = SimpleNamespace(embed='timeF', no_text=False)
    ds = Dataset_TimeMMD(
        args=args, root_path=str(root_path), data_path=data_file,
        flag='test',
        size=[seq_len, 48, pred_len],
        features='S', target='OT', scale=True, timeenc=1, freq='m',
    )
    loader = DataLoader(ds, batch_size=min(8, len(ds)),
                        shuffle=False, num_workers=0)
    return model, loader, device, seq_len, inference_token_len


def _prepare_batch(batch_x, batch_y, tids, amask, ttids, device):
    """Flatten batch for Aurora's expected input layout."""
    from einops import rearrange
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    tids = tids.to(device)
    amask = amask.to(device)
    ttids = ttids.to(device)
    n_vars = batch_x.shape[-1]
    bx_flat = rearrange(batch_x, 'b l c -> (b c) l')
    return (bx_flat, batch_y,
            tids.repeat(n_vars, 1),
            amask.repeat(n_vars, 1),
            ttids.repeat(n_vars, 1))


# =============================================================================
#  Main probing routine (one cell)
# =============================================================================

def probe_one_cell(condition, seed, domain, pred_len,
                    n_probe_batches=3,
                    do_gradnorm=True,
                    do_attention=True,
                    do_divergence=True):
    """Returns a dict of probe statistics for one (cond, seed, domain, h) cell."""
    model, loader, device, seq_len, itl = load_aurora_model_and_dataset(
        condition, seed, domain, pred_len)

    out = {
        'condition': condition,
        'seed': seed,
        'domain': domain,
        'pred_len': pred_len,
        'seq_len': seq_len,
    }

    # ----- Probe A + B -----
    if do_gradnorm or do_attention:
        gn = GradNormProbe() if do_gradnorm else None
        ab = AttentionProbe() if do_attention else None

        # Hook points (Aurora's architecture, validated against repo source):
        #   Probe A: TextEncoder module — hook captures its OUTPUT tensor
        #            (shape [B, num_distill, hidden]). Gradients flow into
        #            this tensor via the trainable projection + cross_text
        #            inside TextEncoder, and downstream via TextGuider.
        #   Probe B: TextGuider (an AuroraAttention) — hook captures the
        #            pre-softmax attention scores; we softmax inside the probe.
        try:
            if gn:
                gn.attach(_find_text_encoder(model))
            if ab:
                ab.attach(_find_text_guider(model))
        except Exception as e:
            out['hook_attach_error'] = f'{type(e).__name__}: {e}'

        # Turn on training mode ONLY to enable gradient flow for probe A.
        # We never call optimizer.step() so weights don't update.
        # Model.eval() is restored after this block.
        if do_gradnorm:
            model.train()
        loss_fn = torch.nn.MSELoss()

        n = 0
        for batch in loader:
            if n >= n_probe_batches:
                break
            try:
                bx, by, tids_r, amask_r, ttids_r = _prepare_batch(*batch, device=device)

                out_model = model(
                    input_ids=bx,
                    text_input_ids=tids_r,
                    text_attention_mask=amask_r,
                    text_token_type_ids=ttids_r,
                    # Aurora's forward computes
                    #   predict_token_num = ceil(max_output_length / inference_token_len)
                    # via a non-None path only when labels OR max_output_length
                    # is provided. Probes do not pass labels, so we explicitly
                    # set max_output_length to the cell's pred_len.
                    max_output_length=pred_len,
                    inference_token_len=itl,
                )
                # Aurora returns a dataclass with .logits or .predictions.
                # Use explicit None checks — `or` triggers bool(tensor) which
                # raises for multi-element tensors.
                pred = getattr(out_model, 'logits', None)
                if pred is None:
                    pred = getattr(out_model, 'predictions', None)
                if pred is None and isinstance(out_model, (tuple, list)):
                    pred = out_model[0]
                if pred is None:
                    out['forward_shape_error'] = 'could not extract prediction from model output'
                    break

                if do_gradnorm:
                    # Shape-align pred and by for a loss. Simple: slice to match.
                    p_flat = pred.reshape(-1)
                    target = by[:, -pred_len:, :].reshape(-1)[:p_flat.shape[0]]
                    p_flat = p_flat[:target.shape[0]]
                    loss = loss_fn(p_flat, target)
                    model.zero_grad()
                    loss.backward()
                    gn.record()
                    model.zero_grad()
                n += 1
            except Exception as e:
                out['batch_error'] = f'{type(e).__name__}: {str(e)[:200]}'
                break

        if gn:
            gn.detach()
            out.update(gn.summary())
        if ab:
            ab.detach()
            out.update(ab.summary())
        model.eval()

    # ----- Probe C (output divergence between with-text and without-text) -----
    if do_divergence:
        for batch in loader:
            try:
                bx, by, tids_r, amask_r, ttids_r = _prepare_batch(*batch, device=device)
                with torch.no_grad():
                    p_with = model.generate(
                        inputs=bx,
                        text_input_ids=tids_r,
                        text_attention_mask=amask_r,
                        text_token_type_ids=ttids_r,
                        inference_token_len=itl,
                        max_output_length=pred_len,
                        num_samples=10,
                    ).mean(0)
                    p_without = model.generate(
                        inputs=bx,
                        text_input_ids=None,
                        text_attention_mask=None,
                        text_token_type_ids=None,
                        inference_token_len=itl,
                        max_output_length=pred_len,
                        num_samples=10,
                    ).mean(0)
                sq_diff = ((p_with - p_without) ** 2).flatten(1).mean(dim=1)
                out['divergence_mean_sq'] = float(sq_diff.mean().item())
                out['divergence_max_sq'] = float(sq_diff.max().item())
                out['divergence_n_samples'] = int(sq_diff.shape[0])
            except Exception as e:
                out['divergence_error'] = f'{type(e).__name__}: {str(e)[:200]}'
            break  # one batch is enough for probe C

    return out


# =============================================================================
#  Output-invariance validator
# =============================================================================

def validate_invariance():
    """Run the same forward pass with and without hooks attached, verify
    outputs are bitwise identical. Confirms probes don't change results."""
    print('=== Output-invariance validation ===')
    print('Loading Aurora on a small test cell...')
    try:
        model, loader, device, seq_len, itl = load_aurora_model_and_dataset(
            'C1_original', 2021, 'Economy', 8)
    except Exception as e:
        print(f'FAIL: could not load Aurora: {e}')
        return False

    # Grab one batch deterministically
    batch = next(iter(loader))
    bx, by, tids_r, amask_r, ttids_r = _prepare_batch(*batch, device=device)

    # Set deterministic seeds so forward pass is repeatable
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Run 1: no hooks
    with torch.no_grad():
        pred_1 = model.generate(
            inputs=bx, text_input_ids=tids_r,
            text_attention_mask=amask_r, text_token_type_ids=ttids_r,
            inference_token_len=itl, max_output_length=8, num_samples=5,
        )

    # Attach probes (Probe A hooks TextEncoder output; Probe B hooks
    # TextGuider attention. See class docstrings for rationale.)
    gn = GradNormProbe()
    ab = AttentionProbe()
    gn.attach(_find_text_encoder(model))
    ab.attach(_find_text_guider(model))

    # Run 2: hooks attached. SAME seed.
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    with torch.no_grad():
        pred_2 = model.generate(
            inputs=bx, text_input_ids=tids_r,
            text_attention_mask=amask_r, text_token_type_ids=ttids_r,
            inference_token_len=itl, max_output_length=8, num_samples=5,
        )

    gn.detach()
    ab.detach()

    same = torch.allclose(pred_1, pred_2, atol=0.0, rtol=0.0)
    max_diff = (pred_1 - pred_2).abs().max().item()
    print(f'  bitwise equal: {same}')
    print(f'  max abs diff:  {max_diff}')
    print(f'  pred_1 sample: {pred_1.flatten()[:4].tolist()}')
    print(f'  pred_2 sample: {pred_2.flatten()[:4].tolist()}')

    # attention hook fired during run 2
    print(f'  attention stats captured during run 2: {len(ab.stats)} (should be >= 1)')
    # gradient hook captured a tensor during run 2 but record() wasn't called
    print(f'  grad captured tensor during run 2: {gn.captured is not None} '
          f'(may be None if no_grad context suppressed it)')

    if same:
        print('\n*** INVARIANCE CONFIRMED: hooks do not change outputs ***')
        return True
    else:
        print(f'\n!!! INVARIANCE VIOLATED: max_diff={max_diff} !!!')
        return False


# =============================================================================
#  Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conditions', nargs='+',
                    default=['C1_original', 'C3_shuffled', 'C6_unimodal', 'C8_oracle'])
    ap.add_argument('--domains', nargs='+',
                    default=['Economy', 'Health', 'Energy'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[2021])
    ap.add_argument('--pred_lens', nargs='+', type=int, default=[8])
    ap.add_argument('--n_probe_batches', type=int, default=3)
    ap.add_argument('--no_gradnorm', action='store_true')
    ap.add_argument('--no_attention', action='store_true')
    ap.add_argument('--no_divergence', action='store_true')
    ap.add_argument('--validate_invariance', action='store_true',
                    help='Run only the output-invariance check and exit.')
    args = ap.parse_args()

    global torch
    import torch as _torch
    torch = _torch

    if args.validate_invariance:
        ok = validate_invariance()
        sys.exit(0 if ok else 1)

    PROBES_ROOT.mkdir(parents=True, exist_ok=True)
    total = (len(args.conditions) * len(args.domains)
             * len(args.seeds) * len(args.pred_lens))
    print(f'Running Aurora probes on {total} cells...')

    i = 0
    for cond in args.conditions:
        for dom in args.domains:
            for seed in args.seeds:
                for h in args.pred_lens:
                    i += 1
                    cell_id = f'{cond}/seed{seed}/{dom}_h{h}'
                    print(f'\n[{i}/{total}] {cell_id}')
                    try:
                        stats = probe_one_cell(
                            cond, seed, dom, h,
                            n_probe_batches=args.n_probe_batches,
                            do_gradnorm=not args.no_gradnorm,
                            do_attention=not args.no_attention,
                            do_divergence=not args.no_divergence,
                        )
                        out_path = probe_path(cond, seed, dom, h)
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        out_path.write_text(json.dumps(stats, indent=2, default=str))
                        for k, v in stats.items():
                            if isinstance(v, float):
                                print(f'    {k}: {v:.6f}')
                            elif isinstance(v, int):
                                print(f'    {k}: {v}')
                    except Exception as e:
                        print(f'    FAIL: {type(e).__name__}: {e}')
                        import traceback
                        traceback.print_exc()


if __name__ == '__main__':
    main()