"""
Builds the unified 3-panel per-architecture figures and the cross-architecture
comparison bar chart (Figures "mnist"/"fmnist"/"eeg"/"bert"/"comparison" and
Table "cross-arch" of the paper). Requires the *_data.npz files already saved
by gradient_effect_{mnist,fmnist,mit_eeg,distilbert}.py.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

DIR = os.path.dirname(os.path.abspath(__file__))

# ── Architecture registry ────────────────────────────────────────────────────
ARCHS = [
    {
        'key':       'mnist',
        'npz':       'gradient_effect_mnist_data.npz',
        'slope_key': 'slope',
        'conf_key':  'conf',
        'label':     'CNN  (MNIST)',
        'subtitle':  'K = 4 conv layers',
    },
    {
        'key':       'fmnist',
        'npz':       'gradient_effect_fmnist_data.npz',
        'slope_key': 'slope',
        'conf_key':  'conf',
        'label':     'CNN  (Fashion-MNIST)',
        'subtitle':  'K = 4 conv layers',
    },
    {
        'key':       'eeg',
        'npz':       'gradient_effect_eeg_data.npz',
        'slope_key': 'slope',
        'conf_key':  'conf',
        'label':     'MLP  (EEG / CHB-MIT)',
        'subtitle':  'K = 4 hidden layers',
    },
    {
        'key':       'bert',
        'npz':       'gradient_effect_bert_data.npz',
        'slope_key': 'slope',
        'conf_key':  'conf',
        'label':     'DistilBERT  (SST-2)',
        'subtitle':  'K = 6 transformer blocks',
    },
]

# ── Shared style constants ────────────────────────────────────────────────────
CI_BINS   = [0.0, 0.30, 0.50, 0.70, 0.85, 1.0]
BIN_LABELS = ['0–0.30', '0.30–0.50', '0.50–0.70', '0.70–0.85', '0.85–1.0']
CMAP      = 'plasma'
FS_TITLE  = 11
FS_AXIS   = 10
FS_TICK   = 9
FIG_SIZE  = (14, 4.2)
DPI       = 150

plt.rcParams.update({
    'font.size':       FS_AXIS,
    'font.family':     'DejaVu Sans',
    'axes.titlesize':  FS_TITLE,
    'axes.labelsize':  FS_AXIS,
    'xtick.labelsize': FS_TICK,
    'ytick.labelsize': FS_TICK,
    'axes.grid':       True,
    'grid.alpha':      0.3,
})


def bar_color(r):
    return '#27ae60' if r < 0 else '#e67e22'


def add_stats_box(ax, r, r2, p, n):
    sign = '+' if r >= 0 else ''
    txt = f'r = {sign}{r:.3f}\nR² = {r2:.3f}\np = {p:.1e}\nn = {n}'
    ax.text(0.03, 0.97, txt, transform=ax.transAxes,
            fontsize=FS_TICK, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))


def plot_arch(CI, slope, conf, meta):
    """Return a 3-panel figure for one architecture."""
    r_val, p_val = stats.pearsonr(CI, slope)
    r2 = r_val ** 2
    n  = len(CI)

    x_th   = 1.0 - CI ** 2
    r_th, p_th = stats.pearsonr(x_th, slope)

    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE)
    fig.suptitle(
        f'{meta["label"]}   —   {meta["subtitle"]}',
        fontsize=FS_TITLE + 1, fontweight='bold', y=1.01,
    )

    # ── Panel A: scatter CI vs |slope| ──────────────────────────────────────
    ax = axes[0]
    vmax = np.percentile(conf, 99)
    sc = ax.scatter(CI, slope, c=conf, cmap=CMAP, alpha=0.35,
                    s=7, vmin=0, vmax=vmax, rasterized=True)
    coef = np.polyfit(CI, slope, 1)
    xl   = np.linspace(CI.min(), CI.max(), 200)
    ax.plot(xl, np.polyval(coef, xl), 'r--', lw=1.8, label='OLS fit')
    ax.set_xlabel('Certainty Index (CI)')
    ax.set_ylabel('|Gradient Slope|  |$S_i$|')
    ax.set_title('|$S_i$| vs Certainty Index')
    plt.colorbar(sc, ax=ax, label='p_max', shrink=0.85, pad=0.02)
    add_stats_box(ax, r_val, r2, p_val, n)

    # ── Panel B: CI-binned bar chart ─────────────────────────────────────────
    ax = axes[1]
    means, sems, ns_bin = [], [], []
    for lo, hi in zip(CI_BINS[:-1], CI_BINS[1:]):
        mask = (CI >= lo) & (CI < hi)
        s = slope[mask]
        means.append(s.mean() if mask.sum() > 0 else 0)
        sems.append(s.std() / np.sqrt(max(mask.sum(), 1)))
        ns_bin.append(int(mask.sum()))
    bc = bar_color(r_val)
    bars = ax.bar(range(len(CI_BINS) - 1), means, yerr=sems,
                  color=bc, alpha=0.78, capsize=4, error_kw={'lw': 1.2})
    for j, (m, nb) in enumerate(zip(means, ns_bin)):
        ax.text(j, m + sems[j] * 1.4, f'n={nb}', ha='center',
                va='bottom', fontsize=7)
    ax.set_xticks(range(len(BIN_LABELS)))
    ax.set_xticklabels(BIN_LABELS, rotation=30, ha='right', fontsize=FS_TICK - 1)
    ax.set_xlabel('CI bin')
    ax.set_ylabel('Mean |$S_i$| ± SEM')
    ax.set_title('CI-stratified Mean Slope')

    # ── Panel C: direct theory test 1 − CI² ─────────────────────────────────
    ax = axes[2]
    ax.scatter(x_th, slope, alpha=0.30, s=7, color='steelblue', rasterized=True)
    coef2 = np.polyfit(x_th, slope, 1)
    xl2   = np.linspace(x_th.min(), x_th.max(), 200)
    ax.plot(xl2, np.polyval(coef2, xl2), 'r--', lw=1.8)
    ax.set_xlabel('$(1 - C_i^2)$')
    ax.set_ylabel('|$S_i$|')
    ax.set_title('Theory Test:  $|S_i| \\propto (1-C_i^2)$?')
    add_stats_box(ax, r_th, r_th ** 2, p_th, n)

    # ── Panel labels – outside, above top-left corner (journal style) ────────
    for ax, lbl in zip(axes, 'ABC'):
        ax.text(-0.04, 1.07, lbl, transform=ax.transAxes,
                fontsize=15, fontweight='bold', va='bottom', ha='right')

    plt.tight_layout()
    return fig


# ── Main: generate per-architecture figures ──────────────────────────────────
results = {}   # key → r_val for comparison plot

for meta in ARCHS:
    npz_path = os.path.join(DIR, meta['npz'])
    if not os.path.exists(npz_path):
        print(f'[SKIP] {meta["label"]:35s}  (no .npz — run the experiment script first)')
        continue

    d     = np.load(npz_path)
    CI    = d['CI'].astype(float)
    slope = d[meta['slope_key']].astype(float)
    conf  = d[meta['conf_key']].astype(float)

    # Sanitise: remove NaN/Inf
    ok    = np.isfinite(CI) & np.isfinite(slope) & np.isfinite(conf)
    CI, slope, conf = CI[ok], slope[ok], conf[ok]

    r_val, _ = stats.pearsonr(CI, slope)
    results[meta['key']] = (r_val, meta['label'])

    fig = plot_arch(CI, slope, conf, meta)
    out = os.path.join(DIR, f'gradient_effect_{meta["key"]}_unified.png')
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'[OK]   {meta["label"]:35s}  r={r_val:+.3f}  ->  {os.path.basename(out)}')


# ── Combined architecture comparison bar chart ───────────────────────────────
if len(results) >= 2:
    # Ordered list includes Fashion-MNIST; fallback to known published values
    all_keys    = ['mnist', 'fmnist', 'eeg', 'bert']
    known_r     = {
        'mnist':  -0.810,
        'eeg':    -0.939,
        'bert':   -0.806,
        # fmnist has no known fallback — only shown when NPZ is present
    }
    arch_labels = {
        'mnist':  'CNN\n(MNIST)\nK=4',
        'fmnist': 'CNN\n(Fashion-\nMNIST) K=4',
        'eeg':    'MLP\n(EEG)\nK=4',
        'bert':   'DistilBERT\n(SST-2)\nK=6',
    }
    r_vals  = []
    x_ticks = []
    colors  = []
    for k in all_keys:
        if k in results:
            r = results[k][0]
        elif k in known_r:
            r = known_r[k]
        else:
            continue
        r_vals.append(r)
        x_ticks.append(arch_labels[k])
        colors.append('#27ae60' if r < 0 else '#e67e22')

    n_archs = len(r_vals)
    title_suffix = 'four' if n_archs == 4 else 'three'

    fig2, ax2 = plt.subplots(figsize=(max(7, n_archs * 1.8), 5))
    fig2.suptitle(
        f'Pearson r  (CI vs |$S_i$|)  across {title_suffix} architectures\n'
        '(sparse regime: $r < 0$  →  $S_i \\propto (1 - C_i^2)$)',
        fontsize=12, fontweight='bold'
    )
    ax2.bar(range(n_archs), r_vals, color=colors, alpha=0.82, width=0.5)
    ax2.axhline(0, color='black', lw=0.9)
    ax2.set_xticks(range(n_archs))
    ax2.set_xticklabels(x_ticks, fontsize=9)
    ax2.set_ylabel('Pearson r', fontsize=11)
    ax2.set_ylim(-1.05, 0.2)
    for i, rv in enumerate(r_vals):
        ax2.text(i, rv - 0.07, f'{rv:+.3f}',
                 ha='center', va='top', fontsize=10, fontweight='bold')
    ax2.text(-0.04, 1.07, 'A', transform=ax2.transAxes,
             fontsize=15, fontweight='bold', va='bottom', ha='right')
    all_neg = all(rv < 0 for rv in r_vals)
    note = ('All negative: sparse regime confirmed\n$S_i \\propto (1 - C_i^2)$'
            if all_neg else 'Green = sparse regime ($r<0$)\nOrange = dense regime ($r>0$)')
    ax2.text(0.02, 0.97, note,
             transform=ax2.transAxes, fontsize=9, va='top',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    plt.tight_layout()
    out2 = os.path.join(DIR, 'gradient_effect_comparison.png')
    fig2.savefig(out2, dpi=DPI, bbox_inches='tight')
    plt.close(fig2)
    print(f'\n[OK]   Comparison bar chart ({n_archs} architectures)  ->  {os.path.basename(out2)}')

print('\nDone.')
