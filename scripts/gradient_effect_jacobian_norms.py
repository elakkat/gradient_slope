"""
Empirically validates the mean-field uniform-Jacobian-norm assumption
underlying the paper's ODE derivation (Section 5.8 / Section "theory" /
Figure "jacobian" / Table "jacobian"): measures the per-layer Jacobian
spectral norm rho_l (exact SVD for the EEG MLP; power iteration via
torch.autograd.functional jvp/vjp for the CNNs and DistilBERT -- no dense
Jacobian is ever formed) and checks whether the exponential-decay profile
it predicts actually holds. Requires gradient_effect_{mnist,fmnist,mit_eeg,
distilbert}.py to have already been run (trained weights + .npz outputs
must exist).
"""

import os
import pyarrow, pandas, sklearn  # pre-warm (see gradient_effect_distilbert.py note)

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd.functional import jvp, vjp
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

DIR = os.path.dirname(os.path.abspath(__file__))

import gradient_effect_mnist as M
import gradient_effect_fmnist as FM
import gradient_effect_mit_eeg as E
import gradient_effect_distilbert as B

N_STRATIFIED = 60      # ~20 low / 20 med / 20 high CI samples per architecture
N_POWER_ITERS = 15

RESULTS = {}   # arch -> dict of arrays / stats


def stratify_by_ci(ci_vals, n_per_bin=20, seed=0):
    """Indices spanning low/med/high CI terciles."""
    rng = np.random.default_rng(seed)
    order = np.argsort(ci_vals)
    n = len(ci_vals)
    bins = np.array_split(order, 3)
    idx = []
    for b in bins:
        take = min(n_per_bin, len(b))
        idx.extend(rng.choice(b, size=take, replace=False))
    return np.array(idx)


def power_iteration_spectral_norm(block_fn, x0, n_iters=N_POWER_ITERS):
    """Top singular value of the Jacobian of block_fn at x0, via alternating
    JVP/VJP power iteration (no dense Jacobian formed)."""
    v = torch.randn_like(x0)
    v = v / v.norm()
    for _ in range(n_iters):
        _, Jv = jvp(block_fn, (x0,), (v,))
        _, JTJv = vjp(block_fn, x0, v=Jv)
        JTJv = JTJv[0] if isinstance(JTJv, tuple) else JTJv
        norm = JTJv.norm()
        if norm < 1e-20:
            break
        v = JTJv / norm
    _, Jv = jvp(block_fn, (x0,), (v,))
    sigma = Jv.norm().item() / (v.norm().item() + 1e-20)
    return sigma


# ─────────────────────────────────────────────────────────────────────────
# EEG (exact, closed-form)
# ─────────────────────────────────────────────────────────────────────────

def run_eeg():
    print('\n=== EEG MLP: exact Jacobian spectral norms ===')
    npz = np.load(os.path.join(DIR, 'gradient_effect_eeg_data.npz'))
    CI, X_test = npz['CI'], npz['X_test']
    model = E.MLP.load_weights(os.path.join(DIR, 'eeg_mlp_trained.npz'),
                                in_dim=X_test.shape[1], n_classes=2)

    idx = stratify_by_ci(CI, n_per_bin=N_STRATIFIED // 3)
    n_hidden = model.n_hidden   # 4 monitored hidden layers -> 3 transitions
    rhos = np.zeros((len(idx), n_hidden - 1))
    alpha_profiles = np.zeros((len(idx), n_hidden))

    for row, i in enumerate(idx):
        x = X_test[i:i+1]
        h = x
        zs = []
        for layer in model.layers[:n_hidden]:
            h = layer.forward(h, dropout=False)
            zs.append(layer.z.copy())
        # exact per-sample activation gradient magnitudes (for the log-linearity check)
        alphas = E.compute_layer_gradients(model, x)
        alpha_profiles[row] = [a[0] for a in alphas]

        # rho_l: Jacobian of layer (l+1)'s output w.r.t. layer l's output
        for l in range(n_hidden - 1):
            W_next = model.layers[l + 1].W          # (in, out)
            z_next = zs[l + 1][0]                     # (out,)
            mask = (z_next > 0).astype(float)         # ReLU'
            J = (mask[:, None] * W_next.T)             # (out, in)
            sv = np.linalg.svd(J, compute_uv=False)
            rhos[row, l] = sv[0]

    _summarize(RESULTS, 'EEG', CI[idx], rhos, alpha_profiles)


# ─────────────────────────────────────────────────────────────────────────
# MNIST / FMNIST CNN (power iteration)
# ─────────────────────────────────────────────────────────────────────────

def make_cnn_block_fns(model):
    """Per-block forward functions matching the model's own forward(), in eval
    mode (BN uses running stats, dropout off) so each is a clean, deterministic
    differentiable map for Jacobian purposes."""
    model.eval()

    def block1(x): return model.pool(F.relu(model.bn1(model.conv1(x))))
    def block2(x): return model.pool(F.relu(model.bn2(model.conv2(x))))
    def block3(x): return F.relu(model.bn3(model.conv3(x)))
    def block4(x): return F.relu(model.bn4(model.conv4(x)))   # dropout excluded (eval-mode identity)
    return [block1, block2, block3, block4]


def run_cnn(name, script_mod, weights_file, norm, dataset_cls, root, n=1000):
    print(f'\n=== {name} CNN: power-iteration Jacobian spectral norms ===')
    npz = np.load(os.path.join(DIR, f'gradient_effect_{name.lower()}_data.npz'))
    CI = npz['CI']

    model = script_mod.GradientCNN(num_classes=10)
    model.load_state_dict(torch.load(os.path.join(DIR, weights_file), map_location='cpu'))
    model.eval()

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*norm)])
    ds = dataset_cls(root=root, train=False, download=True, transform=tfm)

    idx = stratify_by_ci(CI, n_per_bin=N_STRATIFIED // 3)
    blocks = make_cnn_block_fns(model)   # [block1..block4], transitions are block(k+1) applied to a_k

    rhos = np.zeros((len(idx), 3))          # 3 transitions among 4 monitored layers
    alpha_profiles = np.zeros((len(idx), 4))
    device = torch.device('cpu')

    for row, i in enumerate(idx):
        img, _ = ds[i]
        x0 = img.unsqueeze(0)

        a1 = blocks[0](x0)
        a2 = blocks[1](a1)
        a3 = blocks[2](a2)
        a4 = blocks[3](a3)
        acts = [a1, a2, a3, a4]

        # rho_l via power iteration for transitions a1->a2, a2->a3, a3->a4
        for l, fn in enumerate([blocks[1], blocks[2], blocks[3]]):
            rhos[row, l] = power_iteration_spectral_norm(fn, acts[l].detach().requires_grad_(False))

        # exact activation-gradient profile for this sample (log-linearity check)
        alphas_b = script_mod.compute_layer_gradients(model, x0, device)
        alpha_profiles[row] = [a[0] for a in alphas_b]

    _summarize(RESULTS, name, CI[idx], rhos, alpha_profiles)


# ─────────────────────────────────────────────────────────────────────────
# DistilBERT (power iteration on transformer blocks)
# ─────────────────────────────────────────────────────────────────────────

def _call_bert_layer(layer_module, h):
    """DistilBERT's TransformerBlock silently drops the batch dimension on
    output when called directly (bypassing the wrapping Transformer.forward)
    for batch size 1. Restore it so chained calls / jvp / vjp see consistent
    3-D (1, seq, hidden) shapes throughout."""
    out = layer_module(h, attention_mask=None)[0]
    return out if out.dim() == 3 else out.unsqueeze(0)


def run_bert(n_samples=30):
    print('\n=== DistilBERT: power-iteration Jacobian spectral norms ===')
    npz = np.load(os.path.join(DIR, 'gradient_effect_bert_data.npz'))
    CI = npz['CI']
    texts, labels = B.load_sst2_validation(200)   # pool to pick a stratified subset from

    # Build a fresh CI array for this smaller pool (cheap: eval-mode forward only)
    pool_ci = []
    for t in texts:
        inputs = B.tokenizer(t, return_tensors='pt', truncation=True, max_length=B.MAX_LENGTH)
        with torch.no_grad():
            probs = torch.softmax(B.model(**inputs).logits, dim=-1)[0]
        pmax = probs.max().item()
        pool_ci.append(2 * pmax - 1)   # binary CI
    pool_ci = np.array(pool_ci)

    idx = stratify_by_ci(pool_ci, n_per_bin=n_samples // 3)
    n_layers = B.N_LAYERS
    rhos = np.zeros((len(idx), n_layers - 1))
    alpha_profiles = np.zeros((len(idx), n_layers))

    for row, i in enumerate(idx):
        text = texts[i]
        inputs = B.tokenizer(text, return_tensors='pt', truncation=True, max_length=B.MAX_LENGTH)

        with torch.no_grad():
            emb = B.model.distilbert.embeddings(inputs['input_ids'])
        h = emb
        hs = [h]
        for layer in B.model.distilbert.transformer.layer:
            with torch.no_grad():
                h = _call_bert_layer(layer, h)
            hs.append(h)

        def make_block_fn(layer_module):
            def fn(h_in):
                return _call_bert_layer(layer_module, h_in)
            return fn

        for l in range(n_layers - 1):
            fn = make_block_fn(B.model.distilbert.transformer.layer[l + 1])
            rhos[row, l] = power_iteration_spectral_norm(fn, hs[l + 1].detach())

        r = B.analyze_sample(text, true_label=0)
        alpha_profiles[row] = r['grad_mags'] if r is not None else np.nan

    valid = ~np.isnan(alpha_profiles).any(axis=1)
    _summarize(RESULTS, 'DistilBERT', pool_ci[idx][valid], rhos[valid], alpha_profiles[valid])


# ─────────────────────────────────────────────────────────────────────────
# Shared summary / reporting
# ─────────────────────────────────────────────────────────────────────────

def _summarize(results_dict, name, ci_vals, rhos, alpha_profiles):
    mean_rho = rhos.mean(axis=0)
    std_rho = rhos.std(axis=0)
    cov_rho = std_rho / (mean_rho + 1e-12)          # coefficient of variation per transition
    overall_cov = rhos.std() / (rhos.mean() + 1e-12)  # how far from "constant rho" overall

    # correlation of mean-per-sample rho with CI
    mean_rho_per_sample = rhos.mean(axis=1)
    r_rho_ci, p_rho_ci = stats.pearsonr(ci_vals, mean_rho_per_sample)

    # goodness-of-fit of log(alpha_k) ~ k (Eq. 7-8 exponential/geometric assumption)
    k = np.arange(1, alpha_profiles.shape[1] + 1)
    r2_loglinear = []
    for row in range(alpha_profiles.shape[0]):
        a = alpha_profiles[row]
        if np.any(a <= 0):
            r2_loglinear.append(np.nan)
            continue
        slope, intercept, r, p, se = stats.linregress(k, np.log(a))
        r2_loglinear.append(r ** 2)
    r2_loglinear = np.array(r2_loglinear)

    results_dict[name] = dict(
        mean_rho=mean_rho, std_rho=std_rho, cov_rho=cov_rho,
        overall_cov=overall_cov, r_rho_ci=r_rho_ci, p_rho_ci=p_rho_ci,
        r2_loglinear_mean=np.nanmean(r2_loglinear), r2_loglinear_std=np.nanstd(r2_loglinear),
        rhos=rhos, ci_vals=ci_vals, alpha_profiles=alpha_profiles,
    )

    print(f'  Per-transition mean rho: {np.round(mean_rho, 4)}')
    print(f'  Per-transition std  rho: {np.round(std_rho, 4)}')
    print(f'  Overall coefficient of variation (rho across layers & samples): {overall_cov:.3f}')
    print(f'  Corr(mean rho per sample, CI): r={r_rho_ci:.4f}, p={p_rho_ci:.2e}')
    print(f'  log(alpha_k) ~ k goodness-of-fit: mean R^2={np.nanmean(r2_loglinear):.4f} '
          f'(std={np.nanstd(r2_loglinear):.4f})  [tests Eq. 7-8 exponential-decay assumption]')


def plot_summary():
    archs = list(RESULTS.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for name in archs:
        r = RESULTS[name]
        transitions = np.arange(1, len(r['mean_rho']) + 1)
        ax.errorbar(transitions, r['mean_rho'], yerr=r['std_rho'], marker='o', capsize=4, label=name)
    ax.set_xlabel('Layer transition index')
    ax.set_ylabel('Measured Jacobian spectral norm (rho_l)')
    ax.set_title('Is rho_l really constant across layers?\n(Eq. 7-8 mean-field assumption)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    r2_means = [RESULTS[n]['r2_loglinear_mean'] for n in archs]
    r2_stds = [RESULTS[n]['r2_loglinear_std'] for n in archs]
    ax.bar(archs, r2_means, yerr=r2_stds, capsize=5, color='steelblue', alpha=0.8)
    ax.set_ylabel('R^2 of log(alpha_k) ~ k (per-sample), mean +/- std')
    ax.set_title('Goodness-of-fit of the exponential-decay model\n(Eq. 8b)')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(DIR, 'gradient_effect_jacobian_results.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved -> {out}')


def main():
    run_eeg()
    run_cnn('MNIST', M, 'mnist_cnn_trained.pth',
             ((0.1307,), (0.3081,)), torchvision.datasets.MNIST, M.MNIST_ROOT)
    run_cnn('FMNIST', FM, 'fmnist_cnn_trained.pth',
             ((FM.FMNIST_MEAN,), (FM.FMNIST_STD,)), torchvision.datasets.FashionMNIST, FM.FMNIST_ROOT)
    run_bert()

    print('\n' + '=' * 78)
    print('SUMMARY: Jacobian spectral norm validation')
    print('=' * 78)
    for name, r in RESULTS.items():
        print(f'{name:<10} overall CoV(rho)={r["overall_cov"]:.3f}  '
              f'corr(rho,CI) r={r["r_rho_ci"]:+.3f} (p={r["p_rho_ci"]:.2e})  '
              f'log-linear R^2={r["r2_loglinear_mean"]:.3f}')

    npz_path = os.path.join(DIR, 'gradient_effect_jacobian_data.npz')
    np.savez(npz_path, results=np.array(RESULTS, dtype=object))
    print(f'\nSaved -> {npz_path}')

    plot_summary()


if __name__ == '__main__':
    main()
