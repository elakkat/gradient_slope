"""
Trains the MNIST CNN and computes activation-gradient slope vs. Certainty
Index (Section 5.1 / Figure "mnist" / Table "cross-arch" of the paper).
Saves gradient_effect_mnist_data.npz and mnist_cnn_trained.pth for downstream scripts.
"""

import os
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
# Override with an env var to point at an existing torchvision cache.
MNIST_ROOT = os.environ.get('GRADIENT_EFFECT_MNIST_DIR',
                             os.path.join(REPO_ROOT, 'data', 'mnist'))

# ─────────────────────────────────────────────
# 1.  CNN MODEL
# ─────────────────────────────────────────────
class GradientCNN(nn.Module):
    """
    Multi-layer CNN whose intermediate conv activations can be hooked.
    Architecture mirrors a simplified version of the iEEG CNN in the paper.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        # Block 4
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(128)

        self.pool  = nn.MaxPool2d(2, 2)
        self.drop  = nn.Dropout(0.3)

        # After 2 pooling layers: 28 → 14 → 7
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)          # penultimate output (pre-softmax)

        # Keep references to conv layers for gradient hooks
        self.conv_layers = [self.conv1, self.conv2, self.conv3, self.conv4]

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 28 → 14
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 14 → 7
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)   # logits (before softmax)
        return x


# ─────────────────────────────────────────────
# 2.  TRAINING
# ─────────────────────────────────────────────
def train_model(model, train_loader, device, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
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
        acc = 100 * correct / total
        print(f"  Epoch {epoch+1}/{epochs}  loss={total_loss/total:.4f}  acc={acc:.1f}%")
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


# ─────────────────────────────────────────────
# 3.  CERTAINTY INDEX  (Gireesh & Gurupur 2023)
# ─────────────────────────────────────────────
def certainty_index(logits):
    """
    C_i^n = y_i^n - (1/(N-1)) * sum_{j≠i} y_j^n
    where i = argmax (predicted class), n = penultimate layer.
    Range: 0 to (N/(N-1)) * max_logit_gap
    """
    probs = F.softmax(logits, dim=-1).detach().cpu().numpy()   # (B, C)
    ci = np.zeros(probs.shape[0])
    for b in range(probs.shape[0]):
        i    = probs[b].argmax()
        yi   = probs[b, i]
        rest = np.delete(probs[b], i)
        ci[b] = yi - rest.mean()
    return ci   # shape (B,)


# ─────────────────────────────────────────────
# 4.  ACTIVATION GRADIENTS  (Grad-CAM α_k)
# ─────────────────────────────────────────────
def compute_layer_gradients(model, imgs, device):
    """
    Grad-CAM activation gradient weight for each conv layer k:

        alpha_k^C = (1/N) * sum_i |∂p^C / ∂A_k^i|

    where p^C is the SOFTMAX probability for predicted class C,
    A_k is the k-th conv feature map, N = number of spatial elements.

    Using softmax prob (not raw logit) makes the gradient magnitude
    sample-dependent at all layers, naturally correlating with certainty.

    Returns: list of length num_layers, each entry shape (B,)
    """
    import torch.nn.functional as F_nn

    model.eval()
    layer_names = ['conv1', 'conv2', 'conv3', 'conv4']

    # Hooks to capture feature-map gradients (∂p^C/∂A_k)
    feat_grads = {}

    def make_bwd_hook(name):
        def hook(module, grad_in, grad_out):
            feat_grads[name] = grad_out[0].detach()   # (B, C_k, H, W)
        return hook

    hooks = [getattr(model, n).register_backward_hook(make_bwd_hook(n))
             for n in layer_names]

    B = imgs.shape[0]
    layer_alpha = {name: np.zeros(B) for name in layer_names}

    imgs_dev = imgs.to(device)
    logits   = model(imgs_dev)
    probs    = F_nn.softmax(logits, dim=1)
    pred_cls = probs.argmax(dim=1)

    for b in range(B):
        model.zero_grad()
        # Gradient of softmax p^C w.r.t. network params/activations
        p_c = probs[b, pred_cls[b]]
        p_c.backward(retain_graph=True)

        for name in layer_names:
            if name in feat_grads:
                grad_b = feat_grads[name][b]          # (C_k, H, W)
                layer_alpha[name][b] = grad_b.abs().mean().item()

    for h in hooks:
        h.remove()

    return [layer_alpha[n] for n in layer_names]


# ─────────────────────────────────────────────
# 5.  SAMPLE ENTROPY
# ─────────────────────────────────────────────
def sample_entropy(x, m=2, r_factor=0.2):
    """
    SampEn(m, r, N) for a 1-D signal x.
    m = template length, r = tolerance (r_factor * std).
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    r = r_factor * x.std()
    if r == 0:
        return 0.0

    def _count_matches(x, m, r):
        count = 0
        templates = np.array([x[i:i+m] for i in range(N - m)])
        for i in range(N - m):
            dists = np.max(np.abs(templates - templates[i]), axis=1)
            # exclude self-match (j ≠ i)
            count += np.sum(dists < r) - 1
        return count

    A = _count_matches(x, m + 1, r)
    B = _count_matches(x, m, r)
    if B == 0 or A == 0:
        return 0.0
    return -np.log(A / B)


def fast_sample_entropy(x, m=2, r_factor=0.2):
    """
    Vectorised SampEn — faster than the naive version for images flattened to 1-D.
    Uses sliding windows and broadcasts distance computation.
    """
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
        return count / ((N - m) * (N - m - 1))   # normalised

    phi_m   = _phi(m)
    phi_m1  = _phi(m + 1)
    if phi_m == 0:
        return 0.0
    return -np.log(phi_m1 / (phi_m + 1e-10))


def compute_sampen_for_batch(imgs_np, m=2, r_factor=0.2, max_len=64):
    """
    Compute SampEn for a batch of images.
    Each image is flattened to 1-D; we subsample to max_len for speed.
    """
    B = imgs_np.shape[0]
    sampen_vals = np.zeros(B)
    for b in range(B):
        flat = imgs_np[b].flatten()
        # Sub-sample for speed
        if len(flat) > max_len:
            idx  = np.linspace(0, len(flat)-1, max_len, dtype=int)
            flat = flat[idx]
        sampen_vals[b] = fast_sample_entropy(flat, m=m, r_factor=r_factor)
    return sampen_vals


# ─────────────────────────────────────────────
# 6.  SLOPE OF GRADIENTS ACROSS LAYERS
# ─────────────────────────────────────────────
def gradient_slope(layer_alphas):
    """
    layer_alphas: list of arrays, each shape (B,), one per conv layer.
    Returns slope of linear fit (layer_idx → alpha) for each sample.
    Shape: (B,)
    """
    n_layers = len(layer_alphas)
    B        = layer_alphas[0].shape[0]
    x        = np.arange(n_layers, dtype=float)
    slopes   = np.zeros(B)
    for b in range(B):
        y         = np.array([layer_alphas[l][b] for l in range(n_layers)])
        slope, _, _, _, _ = stats.linregress(x, y)
        slopes[b] = slope
    return slopes


# ─────────────────────────────────────────────
# 7.  PLOTTING
# ─────────────────────────────────────────────
def plot_results(layer_alphas, slopes, ci_vals, sampen_vals,
                 pred_labels, true_labels, save_dir='.'):
    """Reproduce the paper's figures using MNIST data."""
    n_layers = len(layer_alphas)
    B        = layer_alphas[0].shape[0]
    layer_idx = np.arange(1, n_layers + 1)

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        'Gradient Effect in CNN — MNIST Replication\n'
        '(Gireesh & Gurupur, 2023 methodology)',
        fontsize=14, fontweight='bold'
    )

    # ── Figure 2 equivalent: activation gradients per layer, each sample a line ──
    ax1 = fig.add_subplot(2, 3, 1)
    colors = plt.cm.viridis(np.linspace(0, 1, B))
    for b in range(B):
        y = [layer_alphas[l][b] for l in range(n_layers)]
        ax1.plot(layer_idx, y, color=colors[b], alpha=0.4, linewidth=0.6)
    ax1.set_xlabel('Conv Layer')
    ax1.set_ylabel('|Activation Gradient| α_k')
    ax1.set_title('Fig 2: Activation Gradients per Layer\n(each line = 1 sample)')
    ax1.set_xticks(layer_idx)

    # ── Figure 3 equivalent: |slope| vs Certainty Index ──
    ax2 = fig.add_subplot(2, 3, 2)
    abs_slopes = np.abs(slopes)
    r, p   = stats.pearsonr(ci_vals, abs_slopes)
    r2     = r ** 2
    m, b_  = np.polyfit(ci_vals, abs_slopes, 1)
    xline  = np.linspace(ci_vals.min(), ci_vals.max(), 200)
    ax2.scatter(ci_vals, abs_slopes, s=10, alpha=0.4, color='steelblue')
    ax2.plot(xline, m * xline + b_, 'r-', linewidth=2)
    ax2.set_xlabel('Certainty Index')
    ax2.set_ylabel('|Slope of Activation Gradients|')
    ax2.set_title(f'Fig 3: Slope vs Certainty Index\nR²={r2:.3f}, p={p:.2e}')

    # ── Figure 4 equivalent: separated by class ──
    unique_classes = np.unique(pred_labels)
    ax3 = fig.add_subplot(2, 3, 3)
    cmap = plt.cm.tab10
    for cls in unique_classes:
        mask = pred_labels == cls
        ax3.scatter(ci_vals[mask], abs_slopes[mask],
                    s=8, alpha=0.5, color=cmap(cls / 10),
                    label=f'Digit {cls}')
    ax3.set_xlabel('Certainty Index')
    ax3.set_ylabel('|Slope of Activation Gradients|')
    ax3.set_title('Fig 4: Slope vs Certainty\nby Predicted Class')
    ax3.legend(fontsize=6, ncol=2)

    # ── Figure 6 equivalent: slope vs SampEn ──
    ax4 = fig.add_subplot(2, 3, 4)
    r_s, p_s = stats.pearsonr(sampen_vals, abs_slopes)
    r2_s     = r_s ** 2
    m_s, b_s = np.polyfit(sampen_vals, abs_slopes, 1)
    xline_s  = np.linspace(sampen_vals.min(), sampen_vals.max(), 200)
    ax4.scatter(sampen_vals, abs_slopes, s=10, alpha=0.4, color='darkorange')
    ax4.plot(xline_s, m_s * xline_s + b_s, 'r-', linewidth=2)
    ax4.set_xlabel('Sample Entropy (SampEn)')
    ax4.set_ylabel('|Slope of Activation Gradients|')
    ax4.set_title(f'Fig 6: Slope vs Sample Entropy\nR²={r2_s:.3f}, p={p_s:.2e}')

    # ── Distribution of certainty index ──
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(ci_vals, bins=40, color='steelblue', edgecolor='white')
    ax5.set_xlabel('Certainty Index')
    ax5.set_ylabel('Count')
    ax5.set_title('Distribution of Certainty Index')

    # ── Distribution of slopes ──
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.hist(abs_slopes, bins=40, color='coral', edgecolor='white')
    ax6.set_xlabel('|Slope of Activation Gradients|')
    ax6.set_ylabel('Count')
    ax6.set_title('Distribution of |Gradient Slopes|')

    plt.tight_layout()
    out_path = f'{save_dir}/gradient_effect_results.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")

    # ── Additional: mean activation gradient per layer ──
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle('Mean Activation Gradient Per Layer', fontsize=13)

    mean_alpha = np.array([layer_alphas[l].mean() for l in range(n_layers)])
    std_alpha  = np.array([layer_alphas[l].std()  for l in range(n_layers)])

    axes[0].bar(layer_idx, mean_alpha, yerr=std_alpha, capsize=5,
                color='steelblue', alpha=0.8)
    axes[0].set_xlabel('Conv Layer')
    axes[0].set_ylabel('Mean |Gradient|')
    axes[0].set_title('Mean Activation Gradient per Layer')
    axes[0].set_xticks(layer_idx)

    # Scatter: true vs predicted accuracy coloring
    correct_mask = pred_labels == true_labels
    axes[1].scatter(ci_vals[correct_mask],  abs_slopes[correct_mask],
                    s=10, alpha=0.5, color='green', label='Correct')
    axes[1].scatter(ci_vals[~correct_mask], abs_slopes[~correct_mask],
                    s=10, alpha=0.5, color='red',   label='Incorrect')
    axes[1].set_xlabel('Certainty Index')
    axes[1].set_ylabel('|Slope of Activation Gradients|')
    axes[1].set_title('Correct vs Incorrect Predictions')
    axes[1].legend()

    out_path2 = f'{save_dir}/gradient_effect_extra.png'
    plt.tight_layout()
    plt.savefig(out_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path2}")

    return r2, r2_s


# ─────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────
def main():
    import sys
    print("=" * 60)
    print("Gradient Effect in DNN — MNIST Replication")
    print("=" * 60)

    # Output directory — use this script's location
    save_dir = SCRIPT_DIR

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # ── Data ──
    print("\n[1/5] Loading MNIST...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = torchvision.datasets.MNIST(root=MNIST_ROOT, train=True,
                                          download=True, transform=transform)
    test_ds  = torchvision.datasets.MNIST(root=MNIST_ROOT, train=False,
                                          download=True, transform=transform)

    # Use a subset for speed: 10k train, 1k test (matches paper's 1000 samples)
    train_subset = Subset(train_ds, range(10000))
    test_subset  = Subset(test_ds,  range(1000))

    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_subset,  batch_size=128, shuffle=False, num_workers=2)

    # ── Train ──
    print("\n[2/5] Training CNN...")
    model = GradientCNN(num_classes=10).to(device)
    model = train_model(model, train_loader, device, epochs=8)
    acc   = evaluate_model(model, test_loader, device)

    # ── Compute gradients, certainty, SampEn on test set ──
    print("\n[3/5] Computing activation gradients and certainty index...")
    model.eval()

    all_layer_alphas = [[] for _ in range(4)]   # 4 conv layers
    all_ci           = []
    all_preds        = []
    all_true         = []
    all_imgs_np      = []
    all_probs_full   = []
    all_pixel_std    = []

    batch_size_grad = 32   # smaller batch for gradient computation
    grad_loader     = DataLoader(test_subset, batch_size=batch_size_grad,
                                 shuffle=False, num_workers=0)

    for batch_idx, (imgs, labels) in enumerate(grad_loader):
        sys.stdout.write(
            f"\r  Batch {batch_idx+1}/{len(grad_loader)} ..."
        )
        sys.stdout.flush()

        imgs_dev = imgs.to(device)

        # Forward for certainty index
        with torch.no_grad():
            logits = model(imgs_dev)
        ci = certainty_index(logits)
        all_ci.append(ci)
        all_probs_full.append(F.softmax(logits, dim=-1).detach().cpu().numpy())

        # Predicted classes
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_true.append(labels.numpy())

        # Store raw pixel values (un-normalised) for SampEn
        all_imgs_np.append(imgs.numpy())
        # Confound for partial-correlation test: per-image pixel std (contrast)
        all_pixel_std.append(imgs.numpy().reshape(imgs.shape[0], -1).std(axis=1))

        # Activation gradients per layer
        layer_alphas = compute_layer_gradients(model, imgs, device)
        for l in range(4):
            all_layer_alphas[l].append(layer_alphas[l])

    print()

    # Concatenate
    all_ci    = np.concatenate(all_ci)
    all_preds = np.concatenate(all_preds)
    all_true  = np.concatenate(all_true)
    all_imgs_np = np.concatenate(all_imgs_np)   # (N, 1, 28, 28)
    all_probs_full = np.concatenate(all_probs_full)   # (N, 10)
    all_pixel_std  = np.concatenate(all_pixel_std)    # (N,)

    layer_alphas_full = [np.concatenate(all_layer_alphas[l]) for l in range(4)]

    # ── Save unified data for make_unified_figures.py ──
    slopes_for_npz = gradient_slope(layer_alphas_full)
    alpha_mat      = np.stack([layer_alphas_full[l] for l in range(4)], axis=1)
    # For N=10 classes: p_max = (9*CI + 1) / 10  (linear function of CI)
    conf_full    = (9.0 * all_ci + 1.0) / 10.0
    correct_full = (all_preds == all_true).astype(int)
    npz_path = os.path.join(save_dir, 'gradient_effect_mnist_data.npz')
    np.savez(npz_path, CI=all_ci, slope=np.abs(slopes_for_npz),
             conf=conf_full, correct=correct_full, alpha_layers=alpha_mat,
             probs_full=all_probs_full, confound_pixel_std=all_pixel_std,
             pred=all_preds, true=all_true)
    print(f"  Saved → {npz_path}")

    # Persist trained weights so downstream scripts (baselines, adversarial,
    # Jacobian analysis) reuse THIS exact model rather than retraining a
    # different one -- keeps all reported statistics consistent with one model.
    weights_path = os.path.join(save_dir, 'mnist_cnn_trained.pth')
    torch.save(model.state_dict(), weights_path)
    print(f"  Saved model → {weights_path}")

    # ── Sample Entropy ──
    print("\n[4/5] Computing Sample Entropy (this may take a minute)...")
    sampen_vals = compute_sampen_for_batch(all_imgs_np, m=2, r_factor=0.2, max_len=64)

    # ── Slope of gradients across layers ──
    print("\n[5/5] Computing gradient slopes and plotting...")
    slopes = gradient_slope(layer_alphas_full)

    # ── Print summary statistics ──
    print(f"\n{'─'*50}")
    print(f"  Samples analysed      : {len(all_ci)}")
    print(f"  Test accuracy         : {acc:.2f}%")
    print(f"  Certainty Index range : [{all_ci.min():.4f}, {all_ci.max():.4f}]")
    print(f"  Gradient slope range  : [{slopes.min():.6f}, {slopes.max():.6f}]")
    print(f"  SampEn range          : [{sampen_vals.min():.4f}, {sampen_vals.max():.4f}]")

    r_ci,  p_ci  = stats.pearsonr(all_ci, np.abs(slopes))
    r_se,  p_se  = stats.pearsonr(sampen_vals, np.abs(slopes))

    print(f"\n  Correlation: |Slope| ~ Certainty Index")
    print(f"    Pearson r = {r_ci:.4f},  R² = {r_ci**2:.4f},  p = {p_ci:.3e}")

    print(f"\n  Correlation: |Slope| ~ Sample Entropy")
    print(f"    Pearson r = {r_se:.4f},  R² = {r_se**2:.4f},  p = {p_se:.3e}")

    if r_ci > 0:
        print("\n  ✓ CONFIRMED: Positive correlation between gradient slope "
              "and certainty index (as in paper).")
    else:
        print("\n  ✗ Negative correlation — investigate further.")

    if r_se > 0:
        print("  ✓ CONFIRMED: Positive correlation between gradient slope "
              "and Sample Entropy (as in paper).")
    else:
        print("  ✗ Negative correlation — investigate further.")

    r2_ci, r2_se = plot_results(
        layer_alphas_full, slopes, all_ci,
        sampen_vals, all_preds, all_true,
        save_dir=save_dir
    )

    print(f"\n{'='*60}")
    print(f"  R² (slope ~ certainty) = {r2_ci:.4f}")
    print(f"  R² (slope ~ SampEn)    = {r2_se:.4f}")
    print(f"  (Paper reported R²=0.64 for iEEG data)")
    print(f"{'='*60}")
    print("\nDone. Figures saved to:", save_dir)


if __name__ == '__main__':
    main()
