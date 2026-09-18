"""
Adversarial (FGSM/PGD) stress test of the "slope-anomaly" detector on MNIST
(Section 5.6 / Figures "adversarial","adversarial-sweep" / Table "adv-stress").

NOTE: an earlier draft of the paper proposed this anomaly score as an
adversarial detector beating plain Certainty Index (CI). On the paper's
retrained checkpoint this did NOT replicate -- the anomaly detector was the
weakest of five tested and CI/raw-slope outperformed it. That retraction is
the paper's actual reported result; this script prints the AUCs it measures
each run and states which detector actually won, rather than assuming an
outcome (see run_epsilon_sweep()/the summary block in main() below).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import os, sys, warnings
warnings.filterwarnings('ignore')

from gradient_effect_shared_baselines import (
    compute_roc_auc, best_direction_auc, top2_margin, mc_dropout_variance,
)

DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(DIR, 'mnist_cnn_trained.pth')
REPO_ROOT  = os.path.dirname(DIR)
# Override with an env var to point at an existing torchvision cache.
MNIST_ROOT = os.environ.get('GRADIENT_EFFECT_MNIST_DIR',
                             os.path.join(REPO_ROOT, 'data', 'mnist'))
EPSILON    = 0.3   # FGSM perturbation in normalised pixel space (headline case)
N_SAMPLES  = 500   # clean test samples to evaluate
EPSILON_SWEEP = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]   # item 9: broaden beyond one point estimate
N_MC_PASSES   = 20


# -- 1. MODEL --------------------------------------------------------------
class GradientCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(128)
        self.pool  = nn.MaxPool2d(2, 2)
        self.drop  = nn.Dropout(0.3)
        self.fc1   = nn.Linear(128 * 7 * 7, 256)
        self.fc2   = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# -- 2. TRAIN / LOAD --------------------------------------------------------
def train_model(model, loader, device, epochs=8):
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    sch = optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
    model.train()
    for ep in range(epochs):
        loss_sum, correct, total = 0, 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward(); opt.step()
            loss_sum += loss.item() * imgs.size(0)
            correct  += (out.argmax(1) == labels).sum().item()
            total    += imgs.size(0)
        sch.step()
        print(f"  Epoch {ep+1}/{epochs}  loss={loss_sum/total:.4f}  "
              f"acc={100*correct/total:.1f}%")
    return model


def get_model(device):
    model = GradientCNN().to(device)
    if os.path.exists(MODEL_PATH):
        print(f"  Loading saved model from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("  No saved model found -- training from scratch...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        ds = torchvision.datasets.MNIST(root=MNIST_ROOT, train=True,
                                        download=True, transform=transform)
        loader = DataLoader(Subset(ds, range(20000)), batch_size=128,
                            shuffle=True, num_workers=0)
        model = train_model(model, loader, device)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  Saved model -> {MODEL_PATH}")
    model.eval()
    return model


# -- 3. CERTAINTY INDEX -----------------------------------------------------
def certainty_index(logits):
    probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
    ci = np.zeros(probs.shape[0])
    for b in range(probs.shape[0]):
        i = probs[b].argmax()
        ci[b] = probs[b, i] - np.delete(probs[b], i).mean()
    return ci


# -- 4. ACTIVATION GRADIENT SLOPE ------------------------------------------
LAYER_NAMES = ['conv1', 'conv2', 'conv3', 'conv4']

def compute_gradients_and_slope(model, imgs, device):
    """
    Returns (slopes, ci, layer_alphas) for a batch.
    slopes: (B,)  absolute OLS slope of alpha_k vs layer index
    ci:     (B,)  Certainty Index
    layer_alphas: list of 4 arrays shape (B,)
    """
    model.eval()
    feat_grads = {}

    def make_hook(name):
        def hook(module, grad_in, grad_out):
            feat_grads[name] = grad_out[0].detach()
        return hook

    hooks = [getattr(model, n).register_backward_hook(make_hook(n))
             for n in LAYER_NAMES]

    B = imgs.shape[0]
    layer_alpha = {n: np.zeros(B) for n in LAYER_NAMES}

    imgs_dev = imgs.to(device)
    logits   = model(imgs_dev)
    probs    = F.softmax(logits, dim=1)
    pred_cls = probs.argmax(dim=1)
    ci       = certainty_index(logits)

    for b in range(B):
        model.zero_grad()
        probs[b, pred_cls[b]].backward(retain_graph=(b < B-1))
        for name in LAYER_NAMES:
            if name in feat_grads:
                layer_alpha[name][b] = feat_grads[name][b].abs().mean().item()

    for h in hooks:
        h.remove()

    alphas = [layer_alpha[n] for n in LAYER_NAMES]
    x_idx  = np.arange(len(LAYER_NAMES), dtype=float)
    slopes = np.array([
        stats.linregress(x_idx, [alphas[l][b] for l in range(len(LAYER_NAMES))])[0]
        for b in range(B)
    ])
    return np.abs(slopes), ci, alphas


# -- 5. FGSM ATTACK ---------------------------------------------------------
def fgsm_attack(model, imgs, labels, epsilon, device):
    """
    Fast Gradient Sign Method (Goodfellow et al., 2014).
    Returns adversarial images tensor (same shape as imgs).
    """
    imgs_d  = imgs.to(device).clone().detach().requires_grad_(True)
    labels_d = labels.to(device)
    loss = nn.CrossEntropyLoss()(model(imgs_d), labels_d)
    model.zero_grad()
    loss.backward()
    adv = (imgs_d + epsilon * imgs_d.grad.data.sign()).detach()
    return adv


# -- 6. ROC / AUC -- promoted to gradient_effect_shared_baselines.py, imported above.


# -- 6b. PGD ATTACK (item 9: broaden beyond single-step FGSM) --------------
def pgd_attack(model, imgs, labels, epsilon, device, alpha=None, n_steps=10):
    """
    Projected Gradient Descent, L-infinity, random start (Madry et al. 2018).
    Multi-step, stronger than FGSM at the same epsilon budget.
    """
    if alpha is None:
        alpha = epsilon / 4.0
    imgs_dev   = imgs.to(device)
    labels_dev = labels.to(device)
    # Random start within the epsilon ball
    delta = (torch.rand_like(imgs_dev) * 2 - 1) * epsilon
    adv = (imgs_dev + delta).detach()
    criterion = nn.CrossEntropyLoss()
    for _ in range(n_steps):
        adv.requires_grad_(True)
        loss = criterion(model(adv), labels_dev)
        model.zero_grad()
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        # Project back into the epsilon-ball around the original image
        adv = torch.clamp(adv, imgs_dev - epsilon, imgs_dev + epsilon).detach()
    return adv


def mc_dropout_stochastic_fn(model):
    """Same pattern as gradient_effect_baselines.py: dropout active, BN in eval."""
    model.eval()
    model.drop.train()

    def f(x_batch):
        with torch.no_grad():
            probs = F.softmax(model(x_batch), dim=1).cpu().numpy()
        return probs
    return f


# -- 6c. EPSILON SWEEP (item 9) ---------------------------------------------
def run_epsilon_sweep(model, device, test_loader, predict_slope, clean_slopes, clean_ci):
    """
    Broadens the adversarial validation beyond the single FGSM eps=0.3 point
    estimate: sweeps EPSILON_SWEEP x {FGSM, PGD}, and compares the slope-anomaly
    detector's AUC against CI-alone, margin-alone, and MC-dropout-variance-alone
    (the same battery used in gradient_effect_baselines.py) at every point.
    """
    print(f"\n[Sweep] Collecting clean-side detector scores (margin, MC-dropout var)...")
    clean_imgs_list, clean_probs_list = [], []
    for imgs, labels in test_loader:
        with torch.no_grad():
            probs = F.softmax(model(imgs.to(device)), dim=1).cpu().numpy()
        clean_imgs_list.append(imgs)
        clean_probs_list.append(probs)
    clean_imgs = torch.cat(clean_imgs_list, dim=0)
    clean_probs = np.concatenate(clean_probs_list, axis=0)
    clean_margin = top2_margin(clean_probs)

    stoch = mc_dropout_stochastic_fn(model)
    clean_mc_var = mc_dropout_variance(stoch, clean_imgs, n_passes=N_MC_PASSES)

    results = {}   # (attack, eps) -> {detector: auc}
    for attack_name in ['FGSM', 'PGD']:
        for eps in EPSILON_SWEEP:
            adv_slopes, adv_ci, adv_probs_list, adv_imgs_list = [], [], [], []
            n_fooled_eps = 0
            for imgs, labels in test_loader:
                adv = (fgsm_attack(model, imgs, labels, eps, device) if attack_name == 'FGSM'
                       else pgd_attack(model, imgs, labels, eps, device))
                with torch.no_grad():
                    preds_adv  = model(adv.to(device)).argmax(1).cpu().numpy()
                    preds_orig = model(imgs.to(device)).argmax(1).cpu().numpy()
                fooled = (preds_adv != labels.numpy()) & (preds_orig == labels.numpy())
                if fooled.sum() == 0:
                    continue
                adv_batch = adv[fooled]
                slopes_b, ci_b, _ = compute_gradients_and_slope(model, adv_batch, device)
                with torch.no_grad():
                    probs_b = F.softmax(model(adv_batch.to(device)), dim=1).cpu().numpy()
                adv_slopes.append(slopes_b)
                adv_ci.append(ci_b)
                adv_probs_list.append(probs_b)
                adv_imgs_list.append(adv_batch)
                n_fooled_eps += int(fooled.sum())

            if n_fooled_eps == 0:
                print(f"  {attack_name} eps={eps}: no successful attacks, skipping")
                continue

            adv_slopes = np.concatenate(adv_slopes)
            adv_ci     = np.concatenate(adv_ci)
            adv_probs  = np.concatenate(adv_probs_list)
            adv_imgs   = torch.cat(adv_imgs_list, dim=0)
            adv_margin = top2_margin(adv_probs)
            adv_mc_var = mc_dropout_variance(stoch, adv_imgs, n_passes=N_MC_PASSES)

            anomaly_clean = clean_slopes - predict_slope(clean_ci)
            anomaly_adv   = adv_slopes   - predict_slope(adv_ci)
            labels_bin = np.array([0] * len(clean_slopes) + [1] * len(adv_slopes))

            aucs = {
                'slope_anomaly':  best_direction_auc(np.concatenate([anomaly_clean, anomaly_adv]), labels_bin),
                'raw_slope':      best_direction_auc(np.concatenate([clean_slopes, adv_slopes]), labels_bin),
                'CI':             best_direction_auc(np.concatenate([-clean_ci, -adv_ci]), labels_bin),
                'margin':         best_direction_auc(np.concatenate([-clean_margin, -adv_margin]), labels_bin),
                'mc_dropout_var': best_direction_auc(np.concatenate([clean_mc_var, adv_mc_var]), labels_bin),
            }
            results[(attack_name, eps)] = aucs
            print(f"  {attack_name} eps={eps}: n_fooled={n_fooled_eps}  " +
                  '  '.join(f'{k}={v:.3f}' for k, v in aucs.items()))

    # -- Plot: AUC vs epsilon, one panel per attack type --
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, attack_name in zip(axes, ['FGSM', 'PGD']):
        detectors = ['slope_anomaly', 'raw_slope', 'CI', 'margin', 'mc_dropout_var']
        for det in detectors:
            xs = [eps for (a, eps) in results if a == attack_name]
            ys = [results[(a, eps)][det] for (a, eps) in results if a == attack_name]
            if xs:
                order = np.argsort(xs)
                ax.plot(np.array(xs)[order], np.array(ys)[order], 'o-', label=det)
        ax.axhline(0.5, color='k', ls='--', lw=1, alpha=0.5)
        ax.set_xlabel('Perturbation budget (epsilon)')
        ax.set_ylabel('AUC (adversarial detection)')
        ax.set_title(f'{attack_name} epsilon sweep')
        ax.set_ylim(0.4, 1.02)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(DIR, 'gradient_effect_adversarial_sweep_results.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Sweep figure saved -> {out}")

    npz_path = os.path.join(DIR, 'gradient_effect_adversarial_sweep_data.npz')
    np.savez(npz_path, results=np.array(results, dtype=object))
    print(f"  Sweep data saved -> {npz_path}")
    return results


# -- 7. MAIN ----------------------------------------------------------------
def main():
    print("=" * 62)
    print("  Adversarial Gradient Anomaly Experiment -- MNIST")
    print("  Counter to: 'slope is just the softmax derivative'")
    print("=" * 62)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}  |  FGSM epsilon: {EPSILON}")

    # -- Data --
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_ds = torchvision.datasets.MNIST(root=MNIST_ROOT, train=False,
                                         download=True, transform=transform)
    test_loader = DataLoader(Subset(test_ds, range(N_SAMPLES)),
                             batch_size=50, shuffle=False, num_workers=0)

    # -- Model --
    print("\n[1/4] Loading / training model...")
    model = get_model(device)

    # -- Evaluate clean accuracy --
    correct = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            correct += (model(imgs.to(device)).argmax(1) == labels.to(device)).sum().item()
    print(f"  Clean test accuracy: {100*correct/N_SAMPLES:.1f}%")

    # -- Compute on clean inputs --
    print("\n[2/4] Clean inputs: computing slopes and CI...")
    clean_slopes, clean_ci, clean_alphas = [], [], [[] for _ in range(4)]
    clean_preds, clean_true = [], []

    for imgs, labels in test_loader:
        slopes_b, ci_b, alphas_b = compute_gradients_and_slope(model, imgs, device)
        with torch.no_grad():
            preds_b = model(imgs.to(device)).argmax(1).cpu().numpy()
        clean_slopes.append(slopes_b)
        clean_ci.append(ci_b)
        clean_preds.append(preds_b)
        clean_true.append(labels.numpy())
        for l in range(4):
            clean_alphas[l].append(alphas_b[l])
        sys.stdout.write(f"\r  {len(clean_ci)*50}/{N_SAMPLES} samples processed...")
        sys.stdout.flush()
    print()

    clean_slopes = np.concatenate(clean_slopes)
    clean_ci     = np.concatenate(clean_ci)
    clean_preds  = np.concatenate(clean_preds)
    clean_true   = np.concatenate(clean_true)
    for l in range(4):
        clean_alphas[l] = np.concatenate(clean_alphas[l])

    # Keep only CORRECTLY classified clean samples for regression fitting
    correct_mask = clean_preds == clean_true
    clean_s_corr = clean_slopes[correct_mask]
    clean_ci_corr = clean_ci[correct_mask]
    r_clean, p_clean = stats.pearsonr(clean_ci_corr, clean_s_corr)
    print(f"  Clean (correct): r={r_clean:.3f}, R^2={r_clean**2:.3f}, "
          f"p={p_clean:.2e}, n={correct_mask.sum()}")

    # Fit clean regression: S = a*(1-CI^2) + offset  [theoretical form]
    X_fit = (1 - clean_ci_corr**2)
    coeffs = np.polyfit(X_fit, clean_s_corr, 1)   # linear in (1-CI^2)
    def predict_slope(ci_vals):
        return coeffs[0] * (1 - ci_vals**2) + coeffs[1]

    # -- Generate adversarial examples --
    print("\n[3/4] Generating FGSM adversarial examples...")
    adv_slopes, adv_ci = [], []
    adv_preds, adv_true_labels = [], []
    adv_alphas = [[] for _ in range(4)]
    adv_imgs_list, orig_imgs_list = [], []

    n_fooled, n_high_ci = 0, 0

    for imgs, labels in test_loader:
        adv = fgsm_attack(model, imgs, labels, EPSILON, device)

        with torch.no_grad():
            preds_adv = model(adv.to(device)).argmax(1).cpu().numpy()
            preds_orig = model(imgs.to(device)).argmax(1).cpu().numpy()

        # Keep adversarials that: (a) fool the model AND (b) were originally correct
        fooled = (preds_adv != labels.numpy()) & (preds_orig == labels.numpy())
        if fooled.sum() == 0:
            continue

        adv_batch   = adv[fooled]
        labels_kept = labels[fooled]
        preds_kept  = preds_adv[fooled]

        slopes_b, ci_b, alphas_b = compute_gradients_and_slope(
            model, adv_batch, device)

        adv_slopes.append(slopes_b)
        adv_ci.append(ci_b)
        adv_preds.append(preds_kept)
        adv_true_labels.append(labels_kept.numpy())
        for l in range(4):
            adv_alphas[l].append(alphas_b[l])
        n_fooled += len(preds_kept)
        n_high_ci += (ci_b > 0.5).sum()

        # Save a few examples for illustration
        if len(adv_imgs_list) < 8:
            for idx in range(min(adv_batch.shape[0], 8 - len(adv_imgs_list))):
                adv_imgs_list.append(adv_batch[idx].cpu())
                orig_imgs_list.append(imgs[fooled][idx].cpu())

        sys.stdout.write(f"\r  Fooled so far: {n_fooled} ...")
        sys.stdout.flush()
    print()

    if n_fooled == 0:
        print("  No adversarial examples generated -- try larger epsilon.")
        return

    adv_slopes = np.concatenate(adv_slopes)
    adv_ci     = np.concatenate(adv_ci)
    adv_preds  = np.concatenate(adv_preds)
    adv_true_labels = np.concatenate(adv_true_labels)
    for l in range(4):
        adv_alphas[l] = np.concatenate(adv_alphas[l])

    attack_rate  = 100 * n_fooled / N_SAMPLES
    high_ci_rate = 100 * n_high_ci / n_fooled
    print(f"  Attack success rate : {attack_rate:.1f}% ({n_fooled}/{N_SAMPLES})")
    print(f"  High-CI adv (CI>0.5): {high_ci_rate:.1f}%")

    r_adv, p_adv = stats.pearsonr(adv_ci, adv_slopes)
    print(f"  Adversarial: r={r_adv:.3f}, R^2={r_adv**2:.3f}, p={p_adv:.2e}")

    # -- Anomaly score --
    # How far above the clean regression line is each sample?
    anomaly_clean = clean_slopes - predict_slope(clean_ci)
    anomaly_adv   = adv_slopes   - predict_slope(adv_ci)

    print(f"\n  Anomaly score (slope - expected from CI):")
    print(f"    Clean  : mean={anomaly_clean.mean():.4e}, "
          f"std={anomaly_clean.std():.4e}")
    print(f"    Advers.: mean={anomaly_adv.mean():.4e}, "
          f"std={anomaly_adv.std():.4e}")

    t_stat, p_ttest = stats.ttest_ind(anomaly_adv, anomaly_clean)
    print(f"  t-test (anomaly adv vs clean): t={t_stat:.2f}, p={p_ttest:.2e}")

    # -- Adversarial detection: ROC / AUC --
    print("\n[4/4] Computing adversarial detection AUC...")
    # Pool clean + adversarial, label 0/1
    all_labels  = np.array([0]*len(clean_slopes) + [1]*len(adv_slopes))

    # Detector 1: gradient slope anomaly (higher = more adversarial)
    all_anomaly = np.concatenate([anomaly_clean, anomaly_adv])
    fpr1, tpr1, auc1 = compute_roc_auc(all_anomaly, all_labels)

    # Detector 2: raw CI-based  (LOWER CI = more suspicious)
    # For adversarials that fool the network, CI is HIGH -> low CI threshold fails
    all_neg_ci  = np.concatenate([-clean_ci, -adv_ci])   # negate: high CI = not adversarial
    fpr2, tpr2, auc2 = compute_roc_auc(-all_neg_ci, all_labels)  # double negate = positive CI
    # Actually: adversarials have HIGH CI (fooled), so using CI ALONE should give AUC < 0.5
    fpr2b, tpr2b, auc2b = compute_roc_auc(all_neg_ci, all_labels)  # low CI = adversarial flag
    auc_ci_detector = max(auc2, auc2b)   # take the better direction

    # Detector 3: raw |slope| alone (without subtracting expected)
    all_slopes_raw = np.concatenate([clean_slopes, adv_slopes])
    fpr3, tpr3, auc3 = compute_roc_auc(all_slopes_raw, all_labels)

    print(f"  AUC -- gradient slope anomaly : {auc1:.3f}")
    print(f"  AUC -- raw |slope| alone      : {auc3:.3f}")
    print(f"  AUC -- softmax CI alone       : {auc_ci_detector:.3f}")
    print(f"  (AUC=0.5 = chance; =1.0 = perfect)")

    # -- FIGURES -----------------------------------------------------------
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "Slope-Anomaly Adversarial Stress Test (FGSM, eps={:.1f})\n"
        "Does the slope-anomaly score detect adversarials beyond CI? "
        "(see printed AUCs for this run's actual answer)".format(EPSILON),
        fontsize=13, fontweight='bold'
    )

    ci_line  = np.linspace(0, 1, 300)
    s_pred   = predict_slope(ci_line)

    # -- Panel A: Clean relationship --
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.scatter(clean_ci_corr, clean_s_corr, s=8, alpha=0.4,
                color='steelblue', label='Clean (correct)')
    ax1.plot(ci_line, s_pred, 'k-', lw=2, label='Fitted Sproportional to(1-C^2)')
    ax1.set_xlabel('Certainty Index (CI)');  ax1.set_ylabel('|Gradient Slope|')
    ax1.set_title(f'Clean inputs\n$R^2$={r_clean**2:.3f}, p={p_clean:.1e}')
    ax1.text(-0.04, 1.07, 'A', transform=ax1.transAxes,
             fontsize=15, fontweight='bold', va='bottom', ha='right')
    ax1.legend(fontsize=9)

    # -- Panel B: Clean + adversarial overlay --
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(clean_ci, clean_slopes, s=8, alpha=0.3,
                color='steelblue', label='Clean')
    ax2.scatter(adv_ci,   adv_slopes,   s=12, alpha=0.6,
                color='crimson', marker='^', label=f'Adversarial (n={n_fooled})')
    ax2.plot(ci_line, s_pred, 'k-', lw=2, label='Clean fit Sproportional to(1-C^2)')
    ax2.set_xlabel('Certainty Index (CI)');  ax2.set_ylabel('|Gradient Slope|')
    ax2.set_title('Clean vs adversarial\nRed = above regression line?')
    ax2.text(-0.04, 1.07, 'B', transform=ax2.transAxes,
             fontsize=15, fontweight='bold', va='bottom', ha='right')
    ax2.legend(fontsize=9)

    # -- Panel C: Anomaly score distributions --
    ax3 = fig.add_subplot(2, 3, 3)
    bins = np.linspace(
        min(anomaly_clean.min(), anomaly_adv.min()),
        max(anomaly_clean.max(), anomaly_adv.max()), 50)
    ax3.hist(anomaly_clean, bins=bins, alpha=0.6, color='steelblue',
             label=f'Clean  mu={anomaly_clean.mean():.2e}', density=True)
    ax3.hist(anomaly_adv,   bins=bins, alpha=0.6, color='crimson',
             label=f'Advers. mu={anomaly_adv.mean():.2e}', density=True)
    ax3.axvline(0, color='k', lw=1.5, ls='--')
    ax3.set_xlabel('Anomaly score  (|S| - predicted S from CI)')
    ax3.set_ylabel('Density')
    ax3.set_title(f'Anomaly score distribution\n'
                  f't={t_stat:.1f}, p={p_ttest:.1e}')
    ax3.text(-0.04, 1.07, 'C', transform=ax3.transAxes,
             fontsize=15, fontweight='bold', va='bottom', ha='right')
    ax3.legend(fontsize=9)

    # -- Panel D: ROC curves --
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(fpr1, tpr1, 'b-',  lw=2,
             label=f'Slope anomaly (AUC={auc1:.3f})')
    ax4.plot(fpr3, tpr3, 'g--', lw=2,
             label=f'|Slope| alone (AUC={auc3:.3f})')
    ax4.plot([0,1],[0,1],  'k:',  lw=1.5, label='Chance (AUC=0.50)')
    ax4.set_xlabel('False Positive Rate'); ax4.set_ylabel('True Positive Rate')
    ax4.set_title('ROC: adversarial detection\nSlope anomaly vs. raw slope')
    ax4.text(-0.04, 1.07, 'D', transform=ax4.transAxes,
             fontsize=15, fontweight='bold', va='bottom', ha='right')
    ax4.legend(fontsize=9)
    ax4.set_xlim([0,1]); ax4.set_ylim([0,1])

    # -- Panel E: Softmax CI is BLIND to adversarials --
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(clean_ci, bins=40, alpha=0.6, color='steelblue',
             label='Clean', density=True)
    ax5.hist(adv_ci,   bins=40, alpha=0.6, color='crimson',
             label='Adversarial', density=True)
    ax5.set_xlabel('Certainty Index (CI)')
    ax5.set_ylabel('Density')
    ax5.set_title(f'Softmax CI distribution\n'
                  f'(AUC of CI-based detector = {auc_ci_detector:.3f})')
    ax5.text(-0.04, 1.07, 'E', transform=ax5.transAxes,
             fontsize=15, fontweight='bold', va='bottom', ha='right')
    ax5.legend(fontsize=9)

    # -- Panel F: Example adversarial images --
    ax6 = fig.add_subplot(2, 3, 6)
    n_show = min(4, len(adv_imgs_list))
    if n_show > 0:
        mean_n = 0.1307; std_n = 0.3081
        for k in range(n_show):
            orig_img = orig_imgs_list[k].squeeze().numpy() * std_n + mean_n
            adv_img  = adv_imgs_list[k].squeeze().numpy()  * std_n + mean_n
            diff     = np.abs(adv_img - orig_img) * 5   # amplified diff
            # mini-subplot via inset
            sub_ax = fig.add_axes([
                0.695 + (k % 2) * 0.12,
                0.14  - (k // 2) * 0.11,
                0.10, 0.09
            ])
            sub_ax.imshow(
                np.concatenate([orig_img, adv_img, diff], axis=1),
                cmap='gray', vmin=0, vmax=1)
            tidx = k
            if tidx < len(adv_true_labels) and tidx < len(adv_preds):
                sub_ax.set_title(
                    f'{adv_true_labels[tidx]}->{adv_preds[tidx]}',
                    fontsize=7)
            sub_ax.axis('off')
        ax6.text(0.5, 0.55,
                 'Example adversarial pairs\n(original | adversarial | diffx5)',
                 ha='center', va='center', fontsize=9, transform=ax6.transAxes)
    ax6.axis('off')
    ax6.set_title('Adversarial examples\n(original → adversarial)')
    ax6.text(-0.04, 1.07, 'F', transform=ax6.transAxes,
             fontsize=15, fontweight='bold', va='bottom', ha='right')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(DIR, 'gradient_effect_adversarial_results.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Figure saved -> {out_path}")

    # -- Print summary for paper --------------------------------------------
    print(f"\n{'='*62}")
    print("  SUMMARY FOR PAPER")
    print(f"{'='*62}")
    print(f"  Architecture      : GradientCNN (4-conv, MNIST)")
    print(f"  Attack            : FGSM, eps={EPSILON}")
    print(f"  Clean samples     : {N_SAMPLES}  (correctly classified)")
    print(f"  Adversarial fooled: {n_fooled} ({attack_rate:.1f}% success rate)")
    print(f"")
    print(f"  Slope ~ CI relationship:")
    print(f"    Clean inputs  : R^2={r_clean**2:.3f}, p={p_clean:.2e}")
    print(f"    Adv inputs    : R^2={r_adv**2:.3f}, p={p_adv:.2e}")
    print(f"")
    print(f"  Anomaly score (|S_i| - predicted S from CI):")
    print(f"    Clean   : {anomaly_clean.mean():.4e} +/- {anomaly_clean.std():.4e}")
    print(f"    Advers. : {anomaly_adv.mean():.4e}  +/- {anomaly_adv.std():.4e}")
    print(f"    t={t_stat:.2f}, p={p_ttest:.2e}  <<< key statistic")
    print(f"")
    print(f"  Adversarial detection AUC:")
    print(f"    Gradient slope anomaly : {auc1:.3f}  <-- proposed detector")
    print(f"    Raw |slope| alone      : {auc3:.3f}")
    print(f"    Softmax CI alone       : {auc_ci_detector:.3f}  <-- baseline (BLIND)")
    print(f"")
    print(f"  Interpretation (computed from THIS run's actual AUCs, not assumed):")
    detector_aucs = {
        'slope anomaly':  auc1,
        'raw |slope|':    auc3,
        'softmax CI':     auc_ci_detector,
    }
    winner = max(detector_aucs, key=detector_aucs.get)
    if winner == 'slope anomaly':
        print(f"    The slope-anomaly score (AUC={auc1:.3f}) outperformed both")
        print(f"    raw |slope| (AUC={auc3:.3f}) and CI alone (AUC={auc_ci_detector:.3f})")
        print(f"    on this run/checkpoint.")
    else:
        print(f"    The slope-anomaly score (AUC={auc1:.3f}) did NOT outperform")
        print(f"    '{winner}' (AUC={detector_aucs[winner]:.3f}) on this run/checkpoint --")
        print(f"    consistent with the paper's own retraction of the anomaly-detector")
        print(f"    claim after retraining (Section 5.6 / Table adv-stress): a single")
        print(f"    favorable checkpoint is not a reliable basis for this claim.")
    print(f"{'='*62}")

    # -- Item 9: broaden beyond the single FGSM eps=0.3 point estimate --------
    print(f"\n{'='*62}")
    print("  EPSILON SWEEP: FGSM + PGD, multiple perturbation budgets")
    print(f"{'='*62}")
    run_epsilon_sweep(model, device, test_loader, predict_slope, clean_slopes, clean_ci)


if __name__ == '__main__':
    main()
