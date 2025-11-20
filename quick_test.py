"""
quick_test.py

Run a single forward pass (E-step + M-step) on a small dummy batch
to check for NaNs/infs, confirm gradient flow, and save diagnostic images.

Usage (PowerShell):
    python quick_test.py

Outputs (in workspace):
    ./quick_test_outputs/
      - reconstructed_0.png
      - original_0.png
      - diagnostics.txt

This script assumes `enhancedlibcem.py` defines `EnhancedLIDECM` and is importable.
"""

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T

# Make sure workspace root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT_DIR = Path('./quick_test_outputs')
OUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_image(tensor, path):
    # tensor expected in range [-1,1] or [0,1]; try to convert sensibly
    t = tensor.detach().cpu()
    if t.ndim == 3:
        t = t
    elif t.ndim == 4:
        t = t[0]
    # try both ranges
    if t.min() >= -1 - 1e-3 and t.max() <= 1 + 1e-3:
        t = (t + 1) / 2
    t = torch.clamp(t, 0, 1)
    img = T.ToPILImage()(t)
    img.save(path)


def main():
    out_log = []
    out_log.append(f"Device: {DEVICE}")

    try:
        from enhancedlibcem import EnhancedLIDECM
    except Exception as e:
        out_log.append(f"ERROR: Could not import enhancedlibcem: {e}")
        (OUT_DIR / 'diagnostics.txt').write_text('\n'.join(out_log))
        print('\n'.join(out_log))
        return

    # init model
    model = EnhancedLIDECM(dataset_name='CIFAR10')
    model.to(DEVICE)
    model.eval()

    out_log.append(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Prepare a tiny dummy batch from random CIFAR-like images
    B = 2
    C = 3
    H = 32
    W = 32

    # Create deterministic pseudo-random images to reproduce
    torch.manual_seed(0)
    dummy = torch.randn(B, C, H, W, device=DEVICE)

    # If model expects normalized inputs, use the same normalization used in training
    # The codebase normalizes with NORM_MEAN and NORM_STD; try common ImageNet mean/std as fallback
    try:
        from enhancedlibcem import NORM_MEAN, NORM_STD
        mean = torch.tensor(NORM_MEAN, device=DEVICE).view(1,3,1,1)
        std = torch.tensor(NORM_STD, device=DEVICE).view(1,3,1,1)
        dummy = dummy * std + mean  # undo random standard normal to [mean,std]
        # Normalize to model input if model expects [-1,1]
        dummy = (dummy - mean) / std
        out_log.append(f"Using NORM_MEAN/NORM_STD from enhancedlibcem: mean={NORM_MEAN}, std={NORM_STD}")
    except Exception:
        out_log.append("NORM_MEAN/NORM_STD not found in enhancedlibcem; using default normalization [-1,1]")

    dummy = dummy.float()
    
    # Enable gradients for m_step optimization
    dummy.requires_grad_(True)
    
    # Save originals for inspection
    try:
        save_image(dummy[0], OUT_DIR / 'original_0.png')
        save_image(dummy[1], OUT_DIR / 'original_1.png')
        out_log.append("Saved original images")
    except Exception as e:
        out_log.append(f"Warning: could not save original images: {e}")

    # Verify gradient setup
    out_log.append(f"Input dummy requires_grad: {dummy.requires_grad}")
    
    # E-step: tokenize (no gradients needed)
    with torch.no_grad():
        try:
            e_out = model.e_step(dummy, current_epoch=0)
            token_indices, ifs_tokens, ifs_probs, patch_info, assignment_confidence = e_out
            out_log.append(f"E-step: token_indices shape: {getattr(token_indices, 'shape', str(type(token_indices)))}")
            out_log.append(f"E-step: mean assignment confidence: {assignment_confidence.mean().item():.6f}")
        except Exception as e:
            out_log.append(f"E-step failure: {e}")
            (OUT_DIR / 'diagnostics.txt').write_text('\n'.join(out_log))
            print('\n'.join(out_log))
            return

    # M-step: requires gradients for backward pass
    try:
        total_loss = model.m_step(dummy, token_indices, ifs_tokens, ifs_probs, patch_info, current_epoch=0)
        out_log.append(f"M-step: returned loss: {total_loss}")
    except Exception as e:
        out_log.append(f"M-step failure: {e}")
        import traceback
        out_log.append(f"Traceback: {traceback.format_exc()}")
        (OUT_DIR / 'diagnostics.txt').write_text('\n'.join(out_log))
        print('\n'.join(out_log))
        return

    # Try to locate reconstructed image file or reconstruction tensor
    # Some code writes reconstructed image within m_step; otherwise, try to call tokenizer.decode
    try:
        # If model produced reconstruction in local scope, it may not be returned. Try decoder fallback.
        from enhancedlibcem import DEVICE as CODE_DEVICE
    except Exception:
        CODE_DEVICE = DEVICE

    try:
        # Try to decode from token indices if available
        recon = None
        try:
            recon = model.tokenizer.decode(token_indices, patch_info)
            if isinstance(recon, torch.Tensor):
                save_image(recon[0].cpu(), OUT_DIR / 'reconstructed_0.png')
                save_image(recon[1].cpu(), OUT_DIR / 'reconstructed_1.png')
                out_log.append('Saved reconstructed images from tokenizer.decode')
        except Exception as e:
            out_log.append(f"tokenizer.decode fallback failed: {e}")

        # If file outputs exist (Colab-style), copy them here
        possible_paths = [Path('/content/inference_results.png'), Path('/content/generated_images_CIFAR10.png')]
        for p in possible_paths:
            if p.exists():
                dest = OUT_DIR / p.name
                try:
                    dest.write_bytes(p.read_bytes())
                    out_log.append(f"Copied existing output: {p}")
                except Exception as e:
                    out_log.append(f"Could not copy {p}: {e}")

    except Exception as e:
        out_log.append(f"Reconstruction saving failed: {e}")

    # EM learner status
    try:
        em_status = model.em_learner.get_status()
        out_log.append(f"EM status: {em_status}")
    except Exception as e:
        out_log.append(f"Could not get EM status: {e}")

    # Parameter and grads check (quick)
    try:
        nan_found = False
        inf_found = False
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                if torch.isnan(p.grad).any():
                    nan_found = True
                    out_log.append(f"NaN in grad: {name}")
                if torch.isinf(p.grad).any():
                    inf_found = True
                    out_log.append(f"Inf in grad: {name}")
        out_log.append(f"NaN in grads: {nan_found}, Inf in grads: {inf_found}")
    except Exception as e:
        out_log.append(f"Grad check failed: {e}")

    # Save diagnostics
    (OUT_DIR / 'diagnostics.txt').write_text('\n'.join(out_log))
    print('\n'.join(out_log))
    print(f"Diagnostics saved to {OUT_DIR / 'diagnostics.txt'}")

if __name__ == '__main__':
    main()
