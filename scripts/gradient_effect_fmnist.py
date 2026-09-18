"""
Trains the Fashion-MNIST CNN (same architecture as gradient_effect_mnist.py)
and computes activation-gradient slope vs. Certainty Index
(Section 5.2 / Figure "fmnist" / Table "cross-arch" of the paper).
Saves gradient_effect_fmnist_data.npz and fmnist_cnn_trained.pth for downstream scripts.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)
# Override with an env var to point at an existing torchvision cache.
FMNIST_ROOT = os.environ.get('GRADIENT_EFFECT_FMNIST_DIR',
                              os.path.join(REPO_ROOT, 'data', 'fmnist'))

FMNIST_CLASSES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Fashion-MNIST channel statistics (train-set moments)
FMNIST_MEAN = 0.2860
FMNIST_STD  = 0.3530


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CNN MODEL  (identical architecture to the MNIST replication)
# ─────────────────────────────────────────────────────────────────────────────
class GradientCNN(nn.Module):
    """4-block conv network; architecture mirrors the iEEG CNN in the paper."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32,  kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.3)

        # 28 → 14 (pool1) → 7 (pool2) → 7×7×128 flat
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.conv_layers = [self.conv1, self.conv2, self.conv3, self.conv4]

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)   # logits (before softmax)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model, train_loader, device, epochs=10):
    """Fashion-MNIST is harder than MNIST so we use 10 epochs."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += imgs.size(0)
        scheduler.step()
        print(f"  Epoch {epoch+1:2d}/{epochs}  "
              f"loss={total_loss/total:.4f}  acc={100*correct/total:.1f}%")
    return model


def evaluate_model(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            correct += (out.argmax(1) == labels).sum().item()
            total   += imgs.size(0)
    acc = 100 * correct / total
    print(f"  Test accuracy: {acc:.2f}%")
    return acc


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CERTAINTY INDEX  (Gireesh & Gurupur 2023)
# ─────────────────────────────────────────────────────────────────────────────
def certainty_index(logits):
    """
    C_i^n = y_i^n − (1/(N−1)) * Σ_{j≠i} y_j^n
    where i = argmax predicted class, y from softmax.
    Range [0, N/(N−1)] ≈ [0, 1.11] for N=10.
    """
    probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
    ci = np.zeros(probs.shape[0])
    for b in range(probs.shape[0]):
        i    = probs[b].argmax()
        ci[b] = probs[b, i] - np.delete(probs[b], i).mean()
    return ci


# ─────────────────────────────────────────────────────────────────────────────
# 4.  ACTIVATION GRADIENTS  (Grad-CAM α_k)
# ─────────────────────────────────────────────────────────────────────────────
def compute_layer_gradients(model, imgs, device):
    """
    α_k = (1/|A_k|) * Σ |∂p^C/∂A_k|
    Using backward hooks on conv layers; softmax probability of predicted class.
    Returns list of 4 arrays, each shape (B,).
    """
    model.eval()
    layer_names = ['conv1', 'conv2', 'conv3', 'conv4']
    feat_grads  = {}

    def make_bwd_hook(name):
        def hook(module, grad_in, grad_out):
            feat_grads[name] = grad_out[0].detach()
        return hook

    hooks = [getattr(model, n).register_backward_hook(make_bwd_hook(n))
             for n in layer_names]

    B            = imgs.shape[0]
    layer_alpha  = {name: np.zeros(B) for name in layer_names}

    imgs_dev = imgs.to(device)
    logits   = model(imgs_dev)
    probs    = F.softmax(logits, dim=1)
    pred_cls = probs.argmax(dim=1)

    for b in range(B):
        model.zero_grad()
        probs[b, pred_cls[b]].backward(retain_graph=True)
        for name in layer_names:
            if name in feat_grads:
                layer_alpha[name][b] = feat_grads[name][b].abs().mean().item()

    for h in hooks:
        h.remove()

    return [layer_alpha[n] for n in layer_names]


# ─────────────────────────────────────────────────────────────────────────────
# 5.  GRADIENT SLOPE ACROSS LAYERS
# ─────────────────────────────────────────────────────────────────────────────
def gradient_slope(layer_alphas):
    """OLS slope of (layer_idx → α_k) for each sample. Returns array (B,)."""
    n_layers = len(layer_alphas)
    B        = layer_alphas[0].shape[0]
    x        = np.arange(n_layers, dtype=float)
    slopes   = np.zeros(B)
    for b in range(B):
        y         = np.array([layer_alphas[l][b] for l in range(n_layers)])
        slope, _, _, _, _ = stats.linregress(x, y)
        slopes[b] = slope
    return slopes


# ─────────────────────────────────────────────────────────────────────────────
# 6.  SAMPLE ENTROPY
# ─────────────────────────────────────────────────────────────────────────────
def fast_sample_entropy(x, m=2, r_factor=0.2):
    x  = np.asarray(x, dtype=float)
    N  = len(x)
    r  = r_factor * (x.std() + 1e-8)

    def _phi(m):
        count = 0
        for i in range(N - m):
            template = x[i:i+m]
            matches  = [x[j:j+m] for j in range(N - m) if j != i]
            if not matches:
                continue
            dists = np.max(np.abs(np.array(matches) - template), axis=1)
            count += (dists < r).sum()
        return count / ((N - m) * (N - m - 1))

    phi_m  = _phi(m)
    phi_m1 = _phi(m + 1)
    if phi_m == 0:
        return 0.0
    return -np.log(phi_m1 / (phi_m + 1e-10))


def compute_sampen_batch(imgs_np, m=2, r_factor=0.2, max_len=64):
    B = imgs_np.shape[0]
    sampen = np.zeros(B)
    for b in range(B):
        flat = imgs_np[b].flatten()
        if len(flat) > max_len:
            idx  = np.linspace(0, len(flat)-1, max_len, dtype=int)
            flat = flat[idx]
        sampen[b] = fast_sample_entropy(flat, m=m, r_factor=r_factor)
    return sampen


# ─────────────────────────────────────────────────────────────────────────────
# 7.  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(layer_alphas, slopes, ci_vals, sampen_vals,
                 pred_labels, true_labels, save_dir='.'):
    n_layers = len(layer_alphas)
    B        = layer_alphas[0].shape[0]
    layer_idx = np.arange(1, n_layers + 1)
    abs_slopes = np.abs(slopes)

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        'Gradient Effect in CNN — Fashion-MNIST\n'
        '(Gireesh & Gurupur 2023 methodology)',
        fontsize=14, fontweight='bold'
    )

    # Panel 1: activation gradients per layer
    ax1 = fig.add_subplot(2, 3, 1)
    colors = plt.cm.viridis(np.linspace(0, 1, B))
    for b in range(B):
        y = [layer_alphas[l][b] for l in range(n_layers)]
        ax1.plot(layer_idx, y, color=colors[b], alpha=0.4, lw=0.6)
    ax1.set_xlabel('Conv Layer')
    ax1.set_ylabel('|Activation Gradient| α_k')
    ax1.set_title('Activation Gradients per Layer\n(each line = 1 sample)')
    ax1.set_xticks(layer_idx)

    # Panel 2: |slope| vs CI
    ax2 = fig.add_subplot(2, 3, 2)
    r, p = stats.pearsonr(ci_vals, abs_slopes)
    m_, b_ = np.polyfit(ci_vals, abs_slopes, 1)
    xl = np.linspace(ci_vals.min(), ci_vals.max(), 200)
    ax2.scatter(ci_vals, abs_slopes, s=10, alpha=0.4, color='steelblue')
    ax2.plot(xl, m_*xl + b_, 'r-', lw=2)
    ax2.set_xlabel('Certainty Index (CI)')
    ax2.set_ylabel('|Gradient Slope|')
    ax2.set_title(f'|Slope| vs CI\nr={r:.3f}, R²={r**2:.3f}, p={p:.2e}')

    # Panel 3: per-class scatter
    ax3 = fig.add_subplot(2, 3, 3)
    cmap_tab10 = plt.cm.tab10
    for cls in range(10):
        mask = pred_labels == cls
        ax3.scatter(ci_vals[mask], abs_slopes[mask],
                    s=8, alpha=0.5, color=cmap_tab10(cls / 10),
                    label=FMNIST_CLASSES[cls])
    ax3.set_xlabel('Certainty Index')
    ax3.set_ylabel('|Gradient Slope|')
    ax3.set_title('Slope vs CI by Class')
    ax3.legend(fontsize=5, ncol=2)

    # Panel 4: |slope| vs SampEn
    ax4 = fig.add_subplot(2, 3, 4)
    r_s, p_s = stats.pearsonr(sampen_vals, abs_slopes)
    m_s, b_s = np.polyfit(sampen_vals, abs_slopes, 1)
    xl_s = np.linspace(sampen_vals.min(), sampen_vals.max(), 200)
    ax4.scatter(sampen_vals, abs_slopes, s=10, alpha=0.4, color='darkorange')
    ax4.plot(xl_s, m_s*xl_s + b_s, 'r-', lw=2)
    ax4.set_xlabel('Sample Entropy')
    ax4.set_ylabel('|Gradient Slope|')
    ax4.set_title(f'|Slope| vs SampEn\nr={r_s:.3f}, R²={r_s**2:.3f}, p={p_s:.2e}')

    # Panel 5: CI distribution
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(ci_vals, bins=40, color='steelblue', edgecolor='white')
    ax5.set_xlabel('Certainty Index')
    ax5.set_ylabel('Count')
    ax5.set_title('Distribution of Certainty Index')

    # Panel 6: slope distribution
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.hist(abs_slopes, bins=40, color='coral', edgecolor='white')
    ax6.set_xlabel('|Gradient Slope|')
    ax6.set_ylabel('Count')
    ax6.set_title('Distribution of |Gradient Slopes|')

    plt.tight_layout()
    out = os.path.join(save_dir, 'gradient_effect_fmnist_results.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out}")

    # Extra: correct vs incorrect + mean layer gradient
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle('Fashion-MNIST — Gradient Analysis (extra)', fontsize=13)

    mean_alpha = np.array([layer_alphas[l].mean() for l in range(n_layers)])
    std_alpha  = np.array([layer_alphas[l].std()  for l in range(n_layers)])
    axes[0].bar(layer_idx, mean_alpha, yerr=std_alpha, capsize=5,
                color='steelblue', alpha=0.8)
    axes[0].set_xlabel('Conv Layer')
    axes[0].set_ylabel('Mean |Gradient|')
    axes[0].set_title('Mean Activation Gradient per Layer')
    axes[0].set_xticks(layer_idx)

    correct_mask = pred_labels == true_labels
    axes[1].scatter(ci_vals[correct_mask],  abs_slopes[correct_mask],
                    s=10, alpha=0.5, color='green', label='Correct')
    axes[1].scatter(ci_vals[~correct_mask], abs_slopes[~correct_mask],
                    s=10, alpha=0.5, color='red',   label='Incorrect')
    axes[1].set_xlabel('Certainty Index')
    axes[1].set_ylabel('|Gradient Slope|')
    axes[1].set_title('Correct vs Incorrect Predictions')
    axes[1].legend()

    out2 = os.path.join(save_dir, 'gradient_effect_fmnist_extra.png')
    plt.tight_layout()
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out2}")

    return r, r_s


# ─────────────────────────────────────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("Gradient Effect in DNN — Fashion-MNIST")
    print("(Same CNN architecture and methodology as MNIST replication)")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # ── Data ──
    print("\n[1/5] Loading Fashion-MNIST...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((FMNIST_MEAN,), (FMNIST_STD,))
    ])
    train_ds = torchvision.datasets.FashionMNIST(
        root=FMNIST_ROOT, train=True,  download=True, transform=transform)
    test_ds  = torchvision.datasets.FashionMNIST(
        root=FMNIST_ROOT, train=False, download=True, transform=transform)

    # 15k training samples; 1k test (matches paper sample size)
    train_subset = Subset(train_ds, range(15000))
    test_subset  = Subset(test_ds,  range(1000))

    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_subset,  batch_size=128, shuffle=False, num_workers=2)

    # ── Train ──
    print("\n[2/5] Training CNN (Fashion-MNIST is harder — 10 epochs)...")
    model = GradientCNN(num_classes=10).to(device)
    model = train_model(model, train_loader, device, epochs=10)
    acc   = evaluate_model(model, test_loader, device)

    # Save trained weights so the adversarial script can load them later
    weights_path = os.path.join(SCRIPT_DIR, 'fmnist_cnn_trained.pth')
    torch.save(model.state_dict(), weights_path)
    print(f"  Saved model → {weights_path}")

    # ── Compute gradients + certainty + SampEn on test set ──
    print("\n[3/5] Computing activation gradients and certainty index...")
    model.eval()

    all_layer_alphas = [[] for _ in range(4)]
    all_ci, all_preds, all_true, all_imgs_np = [], [], [], []
    all_probs_full, all_pixel_std = [], []

    grad_loader = DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=0)

    for batch_idx, (imgs, labels) in enumerate(grad_loader):
        sys.stdout.write(f"\r  Batch {batch_idx+1}/{len(grad_loader)} ...")
        sys.stdout.flush()

        with torch.no_grad():
            logits = model(imgs.to(device))
        ci = certainty_index(logits)
        all_ci.append(ci)
        all_probs_full.append(F.softmax(logits, dim=-1).detach().cpu().numpy())
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_true.append(labels.numpy())
        all_imgs_np.append(imgs.numpy())
        all_pixel_std.append(imgs.numpy().reshape(imgs.shape[0], -1).std(axis=1))

        layer_alphas = compute_layer_gradients(model, imgs, device)
        for l in range(4):
            all_layer_alphas[l].append(layer_alphas[l])

    print()

    all_ci      = np.concatenate(all_ci)
    all_preds   = np.concatenate(all_preds)
    all_true    = np.concatenate(all_true)
    all_imgs_np = np.concatenate(all_imgs_np)
    all_probs_full = np.concatenate(all_probs_full)
    all_pixel_std  = np.concatenate(all_pixel_std)

    layer_alphas_full = [np.concatenate(all_layer_alphas[l]) for l in range(4)]

    # ── Save NPZ for make_unified_figures.py ──
    slopes_npz = gradient_slope(layer_alphas_full)
    alpha_mat  = np.stack([layer_alphas_full[l] for l in range(4)], axis=1)
    conf_full  = (9.0 * all_ci + 1.0) / 10.0          # p_max (N=10 classes)
    correct_full = (all_preds == all_true).astype(int)

    npz_path = os.path.join(SCRIPT_DIR, 'gradient_effect_fmnist_data.npz')
    np.savez(npz_path,
             CI=all_ci,
             slope=np.abs(slopes_npz),
             conf=conf_full,
             correct=correct_full,
             alpha_layers=alpha_mat,
             probs_full=all_probs_full,
             confound_pixel_std=all_pixel_std,
             pred=all_preds, true=all_true)
    print(f"  Saved → {npz_path}")

    # ── Sample Entropy ──
    print("\n[4/5] Computing Sample Entropy...")
    sampen_vals = compute_sampen_batch(all_imgs_np, m=2, r_factor=0.2, max_len=64)

    # ── Slope + figures ──
    print("\n[5/5] Computing gradient slopes and plotting...")
    slopes = gradient_slope(layer_alphas_full)

    # ── Summary statistics ──
    r_ci,  p_ci = stats.pearsonr(all_ci, np.abs(slopes))
    r_se,  p_se = stats.pearsonr(sampen_vals, np.abs(slopes))

    print(f"\n{'─'*60}")
    print(f"  Samples analysed       : {len(all_ci)}")
    print(f"  Test accuracy          : {acc:.2f}%")
    print(f"  CI range               : [{all_ci.min():.4f}, {all_ci.max():.4f}]")
    print(f"  |Slope| range          : [{np.abs(slopes).min():.6f}, {np.abs(slopes).max():.6f}]")
    print(f"  SampEn range           : [{sampen_vals.min():.4f}, {sampen_vals.max():.4f}]")
    print(f"\n  Pearson r (|slope|~CI)     : r={r_ci:+.4f},  R²={r_ci**2:.4f},  p={p_ci:.3e}")
    print(f"  Pearson r (|slope|~SampEn) : r={r_se:+.4f},  R²={r_se**2:.4f},  p={p_se:.3e}")
    print(f"{'─'*60}")

    # Direction check
    print()
    if r_ci < 0:
        print("  SPARSE REGIME (r<0): Higher confidence → LOWER gradient slope.")
        print("  Consistent with MNIST CNN result — both in sparse regime.")
    else:
        print("  Dense regime (r>0): positive slope-CI correlation.")

    r_ci_out, r_se_out = plot_results(
        layer_alphas_full, slopes, all_ci,
        sampen_vals, all_preds, all_true,
        save_dir=SCRIPT_DIR
    )

    print(f"\n{'='*65}")
    print(f"  R  (|slope| ~ CI)      = {r_ci_out:+.4f}  (R²={r_ci_out**2:.4f})")
    print(f"  R  (|slope| ~ SampEn)  = {r_se_out:+.4f}  (R²={r_se_out**2:.4f})")
    print(f"\n  NPZ saved for comparison chart: {npz_path}")
    print(f"{'='*65}")
    print("\nDone. Run make_unified_figures.py to regenerate the 4-arch comparison.")


if __name__ == '__main__':
    main()
