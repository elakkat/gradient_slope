"""
Extends the FGSM slope-anomaly stress test (gradient_effect_adversarial.py)
to the Fashion-MNIST CNN and the CHB-MIT EEG MLP (Section 5.6 / Figure
"adversarial-extended" / Table "adv-stress"). Requires
gradient_effect_fmnist.py and gradient_effect_mit_eeg.py to have already been
run (trained weights + .npz outputs must exist). Prints each architecture's
actual measured AUCs -- see gradient_effect_adversarial.py's docstring for
why this script does not assume the anomaly detector wins.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

DIR = os.path.dirname(os.path.abspath(__file__))

import gradient_effect_fmnist as FM
import gradient_effect_mit_eeg as E
from gradient_effect_shared_baselines import compute_roc_auc, best_direction_auc

EPSILON = 0.3          # same headline perturbation budget as the MNIST script
N_SAMPLES_FMNIST = 500  # matches gradient_effect_adversarial.py's N_SAMPLES

RESULTS = {}   # name -> summary stats dict, printed + saved for the paper


# ─────────────────────────────────────────────────────────────────────────
# Fashion-MNIST: FGSM in normalized pixel space (torch autograd)
# ─────────────────────────────────────────────────────────────────────────

def fgsm_attack_image(model, imgs, labels, epsilon, device):
    imgs_d = imgs.to(device).clone().detach().requires_grad_(True)
    labels_d = labels.to(device)
    loss = nn.CrossEntropyLoss()(model(imgs_d), labels_d)
    model.zero_grad()
    loss.backward()
    return (imgs_d + epsilon * imgs_d.grad.data.sign()).detach()


def run_fmnist(device):
    print('\n=== Fashion-MNIST CNN ===')
    model = FM.GradientCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(
        os.path.join(DIR, 'fmnist_cnn_trained.pth'), map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((FM.FMNIST_MEAN,), (FM.FMNIST_STD,))
    ])
    test_ds = torchvision.datasets.FashionMNIST(
        root=FM.FMNIST_ROOT, train=False, download=True, transform=transform)
    loader = DataLoader(Subset(test_ds, range(N_SAMPLES_FMNIST)),
                         batch_size=50, shuffle=False, num_workers=0)

    # -- Clean pass --
    clean_slopes, clean_ci, clean_preds, clean_true = [], [], [], []
    for imgs, labels in loader:
        alphas = FM.compute_layer_gradients(model, imgs, device)
        clean_slopes.append(np.abs(FM.gradient_slope(alphas)))
        with torch.no_grad():
            logits = model(imgs.to(device))
        clean_ci.append(FM.certainty_index(logits))
        clean_preds.append(logits.argmax(1).cpu().numpy())
        clean_true.append(labels.numpy())
    clean_slopes = np.concatenate(clean_slopes)
    clean_ci     = np.concatenate(clean_ci)
    clean_preds  = np.concatenate(clean_preds)
    clean_true   = np.concatenate(clean_true)
    correct_mask = clean_preds == clean_true
    print(f'  Clean test accuracy: {100*correct_mask.mean():.1f}% (n={N_SAMPLES_FMNIST})')

    # -- FGSM pass --
    adv_slopes, adv_ci = [], []
    n_fooled = 0
    for imgs, labels in loader:
        adv = fgsm_attack_image(model, imgs, labels, EPSILON, device)
        with torch.no_grad():
            preds_adv  = model(adv.to(device)).argmax(1).cpu().numpy()
            preds_orig = model(imgs.to(device)).argmax(1).cpu().numpy()
        fooled = (preds_adv != labels.numpy()) & (preds_orig == labels.numpy())
        if fooled.sum() == 0:
            continue
        adv_batch = adv[fooled]
        alphas = FM.compute_layer_gradients(model, adv_batch, device)
        adv_slopes.append(np.abs(FM.gradient_slope(alphas)))
        with torch.no_grad():
            logits = model(adv_batch.to(device))
        adv_ci.append(FM.certainty_index(logits))
        n_fooled += int(fooled.sum())

    if n_fooled == 0:
        print('  No adversarial examples generated -- try a larger epsilon.')
        return None
    adv_slopes = np.concatenate(adv_slopes)
    adv_ci     = np.concatenate(adv_ci)

    return summarize('Fashion-MNIST CNN',
                      clean_slopes[correct_mask], clean_ci[correct_mask],
                      adv_slopes, adv_ci, N_SAMPLES_FMNIST, n_fooled)


# ─────────────────────────────────────────────────────────────────────────
# EEG: FGSM in z-scored feature space (manual NumPy backprop)
# ─────────────────────────────────────────────────────────────────────────

def compute_input_gradient(model, X, y):
    """dL/dX for cross-entropy loss vs. true labels y, via manual backprop
    through the NumPy MLP. Mirrors the relu_grad/W^T chain that
    E.compute_layer_gradients uses internally, but starts from the loss
    gradient (not the top-class-probability gradient) and continues
    through every layer -- including the first -- to reach the input."""
    logits = model.forward(X, dropout=False)
    _, dL = model.cross_entropy_loss(logits, y)
    da = dL
    for layer in reversed(model.layers):
        dz = da * (E.relu_grad(layer.z) if layer.act == 'relu' else 1.0)
        da = dz @ layer.W.T
    return da


def run_eeg():
    print('\n=== EEG MLP (CHB-MIT) ===')
    npz = np.load(os.path.join(DIR, 'gradient_effect_eeg_data.npz'))
    X_test, y_test = npz['X_test'], npz['true']
    model = E.MLP.load_weights(os.path.join(DIR, 'eeg_mlp_trained.npz'),
                                in_dim=X_test.shape[1], n_classes=2)

    logits = model.predict(X_test)
    preds  = logits.argmax(1)
    correct_mask = preds == y_test
    print(f'  Clean test accuracy: {100*correct_mask.mean():.1f}% (n={len(X_test)})')

    clean_ci = E.certainty_index(logits)
    clean_alphas = E.compute_layer_gradients(model, X_test)
    clean_slopes = np.abs(E.gradient_slope(clean_alphas))

    # -- FGSM --
    grad = compute_input_gradient(model, X_test, y_test)
    X_adv = X_test + EPSILON * np.sign(grad)
    preds_adv = model.predict(X_adv).argmax(1)
    fooled = (preds_adv != y_test) & correct_mask
    n_fooled = int(fooled.sum())
    if n_fooled == 0:
        print('  No adversarial examples generated -- try a larger epsilon.')
        return None

    X_adv_f = X_adv[fooled]
    adv_alphas = E.compute_layer_gradients(model, X_adv_f)
    adv_slopes = np.abs(E.gradient_slope(adv_alphas))
    adv_ci = E.certainty_index(model.predict(X_adv_f))

    return summarize('EEG MLP (CHB-MIT)',
                      clean_slopes[correct_mask], clean_ci[correct_mask],
                      adv_slopes, adv_ci, len(X_test), n_fooled)


# ─────────────────────────────────────────────────────────────────────────
# Shared scoring: clean-fit regression, anomaly score, t-test, ROC AUC
# ─────────────────────────────────────────────────────────────────────────

def summarize(name, clean_slopes, clean_ci, adv_slopes, adv_ci, n_total, n_fooled):
    r_clean, p_clean = stats.pearsonr(clean_ci, clean_slopes)
    r_adv,   p_adv   = stats.pearsonr(adv_ci, adv_slopes)

    # Fit clean regression: S = a*(1-CI^2) + offset  [theoretical form, Eq. prediction]
    X_fit  = 1 - clean_ci**2
    coeffs = np.polyfit(X_fit, clean_slopes, 1)
    predict_slope = lambda ci: coeffs[0] * (1 - ci**2) + coeffs[1]

    anomaly_clean = clean_slopes - predict_slope(clean_ci)
    anomaly_adv   = adv_slopes   - predict_slope(adv_ci)
    t_stat, p_ttest = stats.ttest_ind(anomaly_adv, anomaly_clean)

    labels = np.array([0] * len(clean_slopes) + [1] * len(adv_slopes))
    auc_anomaly   = best_direction_auc(np.concatenate([anomaly_clean, anomaly_adv]), labels)
    auc_raw_slope = best_direction_auc(np.concatenate([clean_slopes, adv_slopes]), labels)
    auc_ci        = best_direction_auc(np.concatenate([-clean_ci, -adv_ci]), labels)

    attack_rate = 100 * n_fooled / n_total
    print(f'  Attack success rate : {attack_rate:.1f}% ({n_fooled}/{n_total})')
    print(f'  Clean  : r={r_clean:+.3f}  R^2={r_clean**2:.3f}  p={p_clean:.2e}  n={len(clean_slopes)}')
    print(f'  Advers.: r={r_adv:+.3f}  R^2={r_adv**2:.3f}  p={p_adv:.2e}  n={len(adv_slopes)}')
    print(f'  Anomaly score: clean mean={anomaly_clean.mean():.4e}  adv mean={anomaly_adv.mean():.4e}')
    print(f'  t-test (anomaly adv vs clean): t={t_stat:.2f}  p={p_ttest:.2e}')
    print(f'  AUC -- slope anomaly : {auc_anomaly:.3f}')
    print(f'  AUC -- raw |slope|   : {auc_raw_slope:.3f}')
    print(f'  AUC -- CI alone      : {auc_ci:.3f}')

    RESULTS[name] = dict(
        n_total=n_total, n_fooled=n_fooled, attack_rate=attack_rate,
        r_clean=r_clean, R2_clean=r_clean**2, p_clean=p_clean, n_clean=len(clean_slopes),
        r_adv=r_adv, R2_adv=r_adv**2, p_adv=p_adv, n_adv=len(adv_slopes),
        anomaly_clean_mean=float(anomaly_clean.mean()), anomaly_adv_mean=float(anomaly_adv.mean()),
        t_stat=t_stat, p_ttest=p_ttest,
        auc_anomaly=auc_anomaly, auc_raw_slope=auc_raw_slope, auc_ci=auc_ci,
    )
    return dict(name=name, clean_slopes=clean_slopes, clean_ci=clean_ci,
                adv_slopes=adv_slopes, adv_ci=adv_ci, predict_slope=predict_slope,
                anomaly_clean=anomaly_clean, anomaly_adv=anomaly_adv)


# ─────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────

def plot_row(row_axes, data):
    name = data['name']
    r = RESULTS[name]
    ci_line = np.linspace(0, 1, 300)
    s_pred = data['predict_slope'](ci_line)

    ax1, ax2, ax3 = row_axes
    ax1.scatter(data['clean_ci'], data['clean_slopes'], s=10, alpha=0.4,
                color='steelblue', label='Clean (correct)')
    ax1.plot(ci_line, s_pred, 'k-', lw=2, label=r'Fit $S\propto(1-C^2)$')
    ax1.set_xlabel('Certainty Index (CI)'); ax1.set_ylabel('|Gradient Slope|')
    ax1.set_title(f'{name}: clean\n$R^2$={r["R2_clean"]:.3f}, p={r["p_clean"]:.1e}')
    ax1.legend(fontsize=8)

    ax2.scatter(data['clean_ci'], data['clean_slopes'], s=8, alpha=0.25,
                color='steelblue', label='Clean')
    ax2.scatter(data['adv_ci'], data['adv_slopes'], s=16, alpha=0.75,
                color='crimson', marker='^', label=f"Adversarial (n={r['n_fooled']})")
    ax2.plot(ci_line, s_pred, 'k-', lw=2, label='Clean fit')
    ax2.set_xlabel('Certainty Index (CI)'); ax2.set_ylabel('|Gradient Slope|')
    ax2.set_title(f'{name}: clean vs.\\ adversarial')
    ax2.legend(fontsize=8)

    bins = np.linspace(min(data['anomaly_clean'].min(), data['anomaly_adv'].min()),
                        max(data['anomaly_clean'].max(), data['anomaly_adv'].max()), 40)
    ax3.hist(data['anomaly_clean'], bins=bins, alpha=0.6, color='steelblue',
             density=True, label='Clean')
    ax3.hist(data['anomaly_adv'], bins=bins, alpha=0.6, color='crimson',
             density=True, label='Adversarial')
    ax3.axvline(0, color='k', ls='--', lw=1.2)
    ax3.set_xlabel('Anomaly score (|S| - predicted S from CI)'); ax3.set_ylabel('Density')
    ax3.set_title(f"{name}: anomaly\nt={r['t_stat']:.2f}, AUC(anomaly)={r['auc_anomaly']:.3f}")
    ax3.legend(fontsize=8)


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 70)
    print('  Adversarial Gradient Anomaly Extension -- Fashion-MNIST + EEG')
    print(f'  FGSM epsilon = {EPSILON}  |  device = {device}')
    print('  Counter to: "slope is just the softmax derivative", tested on')
    print('  two more architectures beyond the original MNIST CNN result.')
    print('=' * 70)

    fm_data  = run_fmnist(device)
    eeg_data = run_eeg()
    rows = [d for d in (fm_data, eeg_data) if d is not None]

    if not rows:
        print('\nNo results to plot -- both architectures failed to produce '
              'adversarial examples.')
        return

    fig, axes = plt.subplots(len(rows), 3, figsize=(16, 5.2 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]
    fig.suptitle(
        f'Gradient Slope Detects Adversarial Examples (FGSM, eps={EPSILON})\n'
        'Extension of the MNIST adversarial validation to Fashion-MNIST and EEG',
        fontsize=13, fontweight='bold')
    for row_axes, data in zip(axes, rows):
        plot_row(row_axes, data)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(DIR, 'gradient_effect_adversarial_extended_results.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nFigure saved -> {out_path}')

    npz_path = os.path.join(DIR, 'gradient_effect_adversarial_extended_data.npz')
    np.savez(npz_path, results=np.array(RESULTS, dtype=object))
    print(f'Data saved -> {npz_path}')

    print('\n' + '=' * 70)
    print('  SUMMARY FOR PAPER')
    print('=' * 70)
    for name, r in RESULTS.items():
        print(f'{name:20s}  clean R^2={r["R2_clean"]:.3f}  adv R^2={r["R2_adv"]:.3f}  '
              f'attack_rate={r["attack_rate"]:.1f}%')
        print(f'{"":20s}  t={r["t_stat"]:.2f} p={r["p_ttest"]:.2e}  '
              f'AUC(anomaly)={r["auc_anomaly"]:.3f}  AUC(raw slope)={r["auc_raw_slope"]:.3f}  '
              f'AUC(CI)={r["auc_ci"]:.3f}')


if __name__ == '__main__':
    main()
