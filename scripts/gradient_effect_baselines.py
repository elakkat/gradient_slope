"""
Benchmarks the gradient slope against softmax entropy, margin, and MC-dropout
variance for misclassification/OOD detection; computes confound-controlled
partial correlation and the temperature-scaling rank-equivalence check
(Section 5.5/5.7 / Tables "baseline-misclass","baseline-ood","partial-corr",
"temp-scaling"). Requires gradient_effect_{mnist,fmnist,mit_eeg,distilbert}.py
to have already been run (trained weights + .npz outputs must exist).
"""

import os
import pyarrow, pandas, sklearn  # pre-warm (see gradient_effect_distilbert.py note)

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

DIR = os.path.dirname(os.path.abspath(__file__))

from gradient_effect_shared_baselines import (
    softmax_entropy, top2_margin, mc_dropout_variance, compute_roc_auc,
    best_direction_auc, partial_correlation, fit_temperature, temperature_scale_logits,
)

import gradient_effect_mnist as M
import gradient_effect_fmnist as FM
import gradient_effect_mit_eeg as E
import gradient_effect_distilbert as B   # loads the pretrained model on import

RESULTS = {}   # arch -> {task: {detector: auc}}
PARTIAL = {}   # arch -> (r_raw, p_raw, r_partial, p_partial, confound_name)

N_MC_PASSES = 20


# ─────────────────────────────────────────────────────────────────────────
# MNIST / FMNIST (shared image-domain logic, parameterized by module)
# ─────────────────────────────────────────────────────────────────────────

MNIST_NORM  = ((0.1307,), (0.3081,))
FMNIST_NORM = ((FM.FMNIST_MEAN,), (FM.FMNIST_STD,))


def load_image_test_set(dataset_cls, root, norm, n=1000):
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*norm)])
    ds = dataset_cls(root=root, train=False, download=True, transform=tfm)
    imgs = torch.stack([ds[i][0] for i in range(n)])
    labels = torch.tensor([ds[i][1] for i in range(n)])
    return imgs, labels


def mc_dropout_stochastic_fn_image(model):
    """Return f(batch_tensor)->np.array(B,10) using ONE stochastic dropout pass,
    with BatchNorm kept in eval mode (uses running stats, not batch stats)."""
    model.eval()
    model.drop.train()   # only the dropout submodule goes into train mode

    def f(x_batch):
        with torch.no_grad():
            logits = model(x_batch)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs
    return f


def run_image_arch(name, script_mod, weights_file, norm, other_dataset_cls, other_root,
                    own_dataset_cls, own_root, num_classes=10):
    print(f'\n=== {name} ===')
    npz = np.load(os.path.join(DIR, f'gradient_effect_{name.lower()}_data.npz'))
    CI, slope, probs_full = npz['CI'], npz['slope'], npz['probs_full']
    correct, confound = npz['correct'], npz['confound_pixel_std']
    n = len(CI)

    model = script_mod.GradientCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(os.path.join(DIR, weights_file), map_location='cpu'))
    model.eval()

    # Reload the SAME test images the original script used (deterministic, no shuffle)
    imgs, labels = load_image_test_set(own_dataset_cls, own_root, norm, n=n)
    with torch.no_grad():
        acc_check = (model(imgs).argmax(1) == labels).float().mean().item()
    print(f'  Reload sanity check: test accuracy = {acc_check*100:.2f}% '
          f'(should match original run)')

    entropy = softmax_entropy(probs_full)

    # ---- MC-dropout variance on the in-distribution test set ----
    stoch = mc_dropout_stochastic_fn_image(model)
    mc_var_in = mc_dropout_variance(stoch, imgs, n_passes=N_MC_PASSES)

    # ---- Task 1: misclassification detection ----
    wrong = 1 - correct
    RESULTS.setdefault(name, {})['misclassification'] = {
        'slope':   best_direction_auc(slope, wrong),
        'CI':      best_direction_auc(-CI, wrong),
        'entropy': best_direction_auc(entropy, wrong),
        'mc_dropout_var': best_direction_auc(mc_var_in, wrong),
    }

    # ---- Task 2: OOD detection (other dataset's images through THIS model) ----
    ood_imgs, _ = load_image_test_set(other_dataset_cls, other_root, norm, n=n)
    ood_ci = script_mod.certainty_index(model(ood_imgs))
    ood_alphas = script_mod.compute_layer_gradients(model, ood_imgs, torch.device('cpu'))
    ood_slope = np.abs(script_mod.gradient_slope(ood_alphas))
    with torch.no_grad():
        ood_probs = F.softmax(model(ood_imgs), dim=1).numpy()
    ood_entropy = softmax_entropy(ood_probs)
    ood_mc_var = mc_dropout_variance(stoch, ood_imgs, n_passes=N_MC_PASSES)

    labels_ood = np.array([0] * n + [1] * n)   # 0=in-dist, 1=OOD
    RESULTS[name]['ood'] = {
        'slope':   best_direction_auc(np.concatenate([slope, ood_slope]), labels_ood),
        'CI':      best_direction_auc(np.concatenate([-CI, -ood_ci]), labels_ood),
        'entropy': best_direction_auc(np.concatenate([entropy, ood_entropy]), labels_ood),
        'mc_dropout_var': best_direction_auc(np.concatenate([mc_var_in, ood_mc_var]), labels_ood),
    }

    # ---- Partial correlation: S_i ~ CI | pixel_std confound ----
    r_raw, p_raw = stats.pearsonr(CI, slope)
    r_partial, p_partial = partial_correlation(CI, slope, confound)
    PARTIAL[name] = (r_raw, p_raw, r_partial, p_partial, 'pixel_std (image contrast)')

    print(f'  Misclassification AUC : {RESULTS[name]["misclassification"]}')
    print(f'  OOD AUC               : {RESULTS[name]["ood"]}')
    print(f'  Partial corr(S,CI|confound): raw r={r_raw:.4f} -> partial r={r_partial:.4f} '
          f'(p={p_partial:.2e})')

    return model, imgs, CI, slope, probs_full, correct


# ─────────────────────────────────────────────────────────────────────────
# EEG
# ─────────────────────────────────────────────────────────────────────────

def run_eeg():
    print('\n=== EEG ===')
    npz = np.load(os.path.join(DIR, 'gradient_effect_eeg_data.npz'))
    CI, slope, probs_full = npz['CI'], npz['slope'], npz['probs_full']
    correct, confound, X_test = npz['correct'], npz['confound_rms'], npz['X_test']
    n = len(CI)

    model = E.MLP.load_weights(os.path.join(DIR, 'eeg_mlp_trained.npz'),
                                in_dim=X_test.shape[1], n_classes=2)
    logits_check = model.predict(X_test)
    acc_check = (logits_check.argmax(1) == npz['true']).mean()
    print(f'  Reload sanity check: test accuracy = {acc_check*100:.2f}% '
          f'(should match original run)')

    entropy = softmax_entropy(probs_full)   # not genuinely distinct from CI (binary) -- reported anyway

    def stoch(x_batch):
        logits = model.forward(x_batch, dropout=True)
        return E.softmax(logits)

    mc_var_in = mc_dropout_variance(stoch, X_test, n_passes=N_MC_PASSES)

    wrong = 1 - correct
    RESULTS.setdefault('EEG', {})['misclassification'] = {
        'slope':   best_direction_auc(slope, wrong),
        'CI':      best_direction_auc(-CI, wrong),
        'mc_dropout_var': best_direction_auc(mc_var_in, wrong),
    }

    # OOD: Gaussian-noise segments (same shape, ~zero-mean-unit-var like real
    # z-scored segments) vs real test segments
    rng = np.random.default_rng(123)
    noise = rng.normal(0.0, 1.0, size=X_test.shape)
    ood_logits = model.predict(noise)
    ood_ci = E.certainty_index(ood_logits)
    ood_alphas = E.compute_layer_gradients(model, noise)
    ood_slope = np.abs(E.gradient_slope(ood_alphas))
    ood_probs = E.softmax(ood_logits)
    ood_entropy = softmax_entropy(ood_probs)
    ood_mc_var = mc_dropout_variance(stoch, noise, n_passes=N_MC_PASSES)

    labels_ood = np.array([0] * n + [1] * len(noise))
    RESULTS['EEG']['ood'] = {
        'slope':   best_direction_auc(np.concatenate([slope, ood_slope]), labels_ood),
        'CI':      best_direction_auc(np.concatenate([-CI, -ood_ci]), labels_ood),
        'mc_dropout_var': best_direction_auc(np.concatenate([mc_var_in, ood_mc_var]), labels_ood),
    }

    r_raw, p_raw = stats.pearsonr(CI, slope)
    r_partial, p_partial = partial_correlation(CI, slope, confound)
    PARTIAL['EEG'] = (r_raw, p_raw, r_partial, p_partial, 'RMS energy (signal power)')

    print(f'  Misclassification AUC : {RESULTS["EEG"]["misclassification"]}')
    print(f'  OOD AUC               : {RESULTS["EEG"]["ood"]}')
    print(f'  Partial corr(S,CI|confound): raw r={r_raw:.4f} -> partial r={r_partial:.4f} '
          f'(p={p_partial:.2e})')


# ─────────────────────────────────────────────────────────────────────────
# DistilBERT
# ─────────────────────────────────────────────────────────────────────────

N_BERT_MC_SAMPLES = 250   # subsample for the expensive MC-dropout pass


def bert_stochastic_predict_one(text):
    """One stochastic (dropout-active) forward pass for a single text ->
    (2,) softmax prob vector. DistilBERT has no BatchNorm, so full
    model.train() is safe (only activates its internal dropout layers)."""
    B.model.train()
    inputs = B.tokenizer(text, return_tensors='pt', truncation=True,
                          max_length=B.MAX_LENGTH, padding=False)
    inputs = {k: v.to(B.device) for k, v in inputs.items()}
    with torch.no_grad():
        probs = torch.softmax(B.model(**inputs).logits, dim=-1)[0].cpu().numpy()
    B.model.eval()
    return probs


def run_bert():
    print('\n=== DistilBERT ===')
    npz = np.load(os.path.join(DIR, 'gradient_effect_bert_data.npz'))
    CI, slope, probs_full = npz['CI'], npz['slope'], npz['probs_full']
    correct, confound = npz['correct'], npz['confound_n_tokens']
    n = len(CI)

    wrong = 1 - correct
    RESULTS.setdefault('DistilBERT', {})['misclassification'] = {
        'slope': best_direction_auc(slope, wrong),
        'CI':    best_direction_auc(-CI, wrong),
    }

    # ---- MC-dropout on a subsample (expensive: T stochastic passes/text) ----
    print(f'  Computing MC-dropout variance on {N_BERT_MC_SAMPLES} texts '
          f'({N_MC_PASSES} passes each)...')
    texts, labels = B.load_sst2_validation(N_BERT_MC_SAMPLES)
    mc_records = []
    for i, (txt, lbl) in enumerate(zip(texts, labels)):
        r = B.analyze_sample(txt, lbl)   # deterministic (eval mode) CI/slope
        if r is None:
            continue
        passes = np.stack([bert_stochastic_predict_one(txt) for _ in range(N_MC_PASSES)])
        top_cls = passes.mean(axis=0).argmax()
        mc_var = passes[:, top_cls].var()
        mc_records.append({**r, 'mc_var': mc_var})
        if i % 50 == 0:
            print(f'    {i}/{len(texts)}')

    mc_ci = np.array([r['ci'] for r in mc_records])
    mc_slope = np.array([r['abs_slope'] for r in mc_records])
    mc_wrong = 1 - np.array([r['correct'] for r in mc_records])
    mc_var_arr = np.array([r['mc_var'] for r in mc_records])
    RESULTS['DistilBERT']['misclassification_mcsubset'] = {
        'slope': best_direction_auc(mc_slope, mc_wrong),
        'CI':    best_direction_auc(-mc_ci, mc_wrong),
        'mc_dropout_var': best_direction_auc(mc_var_arr, mc_wrong),
    }
    print(f'  (n={len(mc_records)} subset) Misclassification AUC incl. MC-dropout: '
          f'{RESULTS["DistilBERT"]["misclassification_mcsubset"]}')

    # ---- OOD: QQP question sentences (different task domain, same tokenizer) ----
    print('  Loading QQP as OOD text source...')
    try:
        from datasets import load_dataset
        qqp = load_dataset('glue', 'qqp', split='validation')
    except Exception:
        qqp = load_dataset('nyu-mll/glue', 'qqp', split='validation')
    n_ood = min(300, len(qqp))
    ood_texts = [qqp[i]['question1'] for i in range(n_ood)]

    ood_records = []
    for i, txt in enumerate(ood_texts):
        r = B.analyze_sample(txt, true_label=0)   # true_label unused (no 'correct' meaning for OOD)
        if r is not None:
            ood_records.append(r)
    ood_ci = np.array([r['ci'] for r in ood_records])
    ood_slope = np.array([r['abs_slope'] for r in ood_records])

    n_in = len(CI)
    n_out = len(ood_ci)
    labels_ood = np.array([0] * n_in + [1] * n_out)
    RESULTS['DistilBERT']['ood'] = {
        'slope': best_direction_auc(np.concatenate([slope, ood_slope]), labels_ood),
        'CI':    best_direction_auc(np.concatenate([-CI, -ood_ci]), labels_ood),
    }
    print(f'  OOD AUC (SST-2 vs QQP, n_in={n_in}, n_out={n_out}): {RESULTS["DistilBERT"]["ood"]}')

    r_raw, p_raw = stats.pearsonr(CI, slope)
    r_partial, p_partial = partial_correlation(CI, slope, confound)
    PARTIAL['DistilBERT'] = (r_raw, p_raw, r_partial, p_partial, 'token count (sentence length)')
    print(f'  Partial corr(S,CI|confound): raw r={r_raw:.4f} -> partial r={r_partial:.4f} '
          f'(p={p_partial:.2e})')


# ─────────────────────────────────────────────────────────────────────────
# Rank-equivalence sanity check: temperature scaling vs raw CI (item 7 note)
# ─────────────────────────────────────────────────────────────────────────

def temperature_rank_equivalence_check(name, probs_full, correct):
    """log(probs_full) is an exact stand-in for logits for this purpose
    (softmax is shift-invariant; see gradient_effect_shared_baselines docstring).
    Demonstrates directly that scaling by ANY fixed T cannot change the AUC
    ranking of the top-class confidence -- temperature scaling recalibrates
    probability VALUES but is a monotonic (rank-preserving) transform, so it
    cannot help or hurt misclassification/OOD detection AUC. This is why
    temperature scaling is not included as a separate ranking-AUC baseline
    above; fit_temperature()/temperature_scale_logits() are provided in
    gradient_effect_shared_baselines.py for anyone who wants ECE/calibration
    analysis instead (a different, non-ranking question)."""
    logits_proxy = np.log(np.clip(probs_full, 1e-12, 1.0))
    raw_conf = probs_full.max(axis=1)
    for T_test in [0.5, 1.0, 2.0, 5.0]:
        scaled = temperature_scale_logits(logits_proxy, T_test)
        scaled_conf = scaled.max(axis=1)
        rank_corr = stats.spearmanr(raw_conf, scaled_conf).correlation
        wrong = 1 - correct
        auc_raw = best_direction_auc(-raw_conf, wrong)
        auc_scaled = best_direction_auc(-scaled_conf, wrong)
        print(f'  [{name}] T={T_test}: Spearman(raw_conf, T-scaled_conf)={rank_corr:.6f}  '
              f'AUC(raw)={auc_raw:.4f}  AUC(T-scaled)={auc_scaled:.4f}  '
              f'{"MATCH" if abs(auc_raw-auc_scaled) < 1e-9 else "MISMATCH"}')


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────

def main():
    _, _, mnist_ci, mnist_slope, mnist_probs, mnist_correct = run_image_arch(
        'MNIST', M, 'mnist_cnn_trained.pth', MNIST_NORM,
        other_dataset_cls=torchvision.datasets.FashionMNIST, other_root=FM.FMNIST_ROOT,
        own_dataset_cls=torchvision.datasets.MNIST, own_root=M.MNIST_ROOT,
    )
    _, _, fmnist_ci, fmnist_slope, fmnist_probs, fmnist_correct = run_image_arch(
        'FMNIST', FM, 'fmnist_cnn_trained.pth', FMNIST_NORM,
        other_dataset_cls=torchvision.datasets.MNIST, other_root=M.MNIST_ROOT,
        own_dataset_cls=torchvision.datasets.FashionMNIST, own_root=FM.FMNIST_ROOT,
    )
    run_eeg()
    run_bert()

    print('\n=== Rank-equivalence sanity check (temperature scaling) ===')
    print('  (Exact invariance is a BINARY-classifier property -- p_max there is a')
    print('   function of the single top-2 logit margin alone. For N>2 (MNIST), p_max')
    print('   depends on the full logit vector, so T-scaling is only APPROXIMATELY')
    print('   rank-preserving. Checking both regimes:)')
    eeg_npz = np.load(os.path.join(DIR, 'gradient_effect_eeg_data.npz'))
    bert_npz = np.load(os.path.join(DIR, 'gradient_effect_bert_data.npz'))
    temperature_rank_equivalence_check('EEG (binary, exact)', eeg_npz['probs_full'], eeg_npz['correct'])
    temperature_rank_equivalence_check('DistilBERT (binary, exact)', bert_npz['probs_full'], bert_npz['correct'])
    temperature_rank_equivalence_check('MNIST (N=10, approximate)', mnist_probs, mnist_correct)

    # ---- Summary table ----
    print('\n' + '=' * 78)
    print('SUMMARY: AUC by architecture x task x detector')
    print('=' * 78)
    for arch, tasks in RESULTS.items():
        for task, detectors in tasks.items():
            row = '  '.join(f'{k}={v:.3f}' for k, v in detectors.items())
            print(f'{arch:<12}{task:<26}{row}')

    print('\n' + '=' * 78)
    print('SUMMARY: partial correlation S_i ~ CI | confound')
    print('=' * 78)
    for arch, (r_raw, p_raw, r_p, p_p, cname) in PARTIAL.items():
        print(f'{arch:<12} confound={cname:<28} raw r={r_raw:+.4f} (p={p_raw:.1e})  '
              f'-> partial r={r_p:+.4f} (p={p_p:.1e})')

    # ---- Save + plot ----
    npz_path = os.path.join(DIR, 'gradient_effect_baselines_data.npz')
    np.savez(npz_path,
             results=np.array(RESULTS, dtype=object),
             partial=np.array(PARTIAL, dtype=object))
    print(f'\nSaved -> {npz_path}')

    plot_summary()


def plot_summary():
    archs = list(RESULTS.keys())
    tasks = sorted({t for a in RESULTS.values() for t in a.keys()})
    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 5), squeeze=False)
    axes = axes[0]
    for ax, task in zip(axes, tasks):
        detectors = sorted({d for a in RESULTS.values() if task in a for d in a[task].keys()})
        x = np.arange(len(archs))
        width = 0.8 / max(len(detectors), 1)
        for j, det in enumerate(detectors):
            vals = [RESULTS[a].get(task, {}).get(det, np.nan) for a in archs]
            ax.bar(x + j * width, vals, width, label=det)
        ax.set_xticks(x + width * (len(detectors) - 1) / 2)
        ax.set_xticklabels(archs, rotation=20)
        ax.axhline(0.5, color='k', ls='--', lw=1, alpha=0.5, label='chance')
        ax.set_ylim(0.4, 1.0)
        ax.set_ylabel('AUC')
        ax.set_title(task)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(DIR, 'gradient_effect_baselines_results.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved -> {out}')


if __name__ == '__main__':
    main()
