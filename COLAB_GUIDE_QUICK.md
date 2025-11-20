# Colab Setup Guide - Enhanced LIDECM

## Quick Start (5 minutes)

### 1. **Prepare Files** (on your machine)
- Ensure you have: `improved_lidecm.pt` (the pre-trained checkpoint)
- Upload to your Google Drive or keep ready to upload to Colab

### 2. **Clone/Upload Code to Colab**

Copy this entire cell and run in Colab:

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Option A: Copy from Drive (if already uploaded there)
!cp -r /content/drive/MyDrive/enhancedlibcem /content/ 2>/dev/null || echo "Not in Drive root"

# Option B: Clone from GitHub (if repo is public)
# !git clone https://github.com/YOUR-USER/enhancedlibcem /content/enhancedlibcem

# Option C: Upload via Colab UI
# Files → Upload to session → drag enhancedlibcem folder

# Verify code is present
!ls -la /content/enhancedlibcem/quick_test.py
```

### 3. **Upload Checkpoint to Colab**

If checkpoint is not yet in the session:

```python
# Method 1: Copy from Drive (if already uploaded)
!cp /content/drive/MyDrive/improved_lidecm.pt /content/enhancedlibcem/

# Method 2: Upload via UI
# Files → Upload → drag improved_lidecm.pt into /content/enhancedlibcem/

# Verify checkpoint exists
!ls -lh /content/enhancedlibcem/improved_lidecm.pt
```

### 4. **Install Dependencies**

```bash
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q git+https://github.com/openai/CLIP.git numpy matplotlib scikit-image scipy numba pillow tqdm
```

### 5. **Run Quick Diagnostics**

```bash
%cd /content/enhancedlibcem
!python quick_test.py
```

Check the output for:
- ✅ `M-step: returned loss`
- ✅ `Saved reconstructed images`
- ✅ `NaN in grads: False`

### 6. **Run Inference with Checkpoint**

```bash
%cd /content/enhancedlibcem
!python inference_from_checkpoint.py --checkpoint improved_lidecm.pt --outdir /content/inference_out
```

or use the smart loader:

```bash
!python colab_inference.py --outdir /content/inference_out
```

### 7. **View & Save Results**

```python
from IPython.display import Image, display
from pathlib import Path

# Show diagnostic results
with open('/content/enhancedlibcem/quick_test_outputs/diagnostics.txt', 'r') as f:
    print(f.read())

# Show generated images
print("\n=== Quick Test: Original vs Reconstructed ===")
display(Image('/content/enhancedlibcem/quick_test_outputs/original_0.png'))
display(Image('/content/enhancedlibcem/quick_test_outputs/reconstructed_0.png'))

print("\n=== Inference Results ===")
display(Image('/content/inference_out/inference_results.png'))

# Save to Drive
import shutil
shutil.copytree('/content/enhancedlibcem/quick_test_outputs', 
                '/content/drive/MyDrive/enhancedlibcem_results/quick_test', 
                dirs_exist_ok=True)
shutil.copytree('/content/inference_out',
                '/content/drive/MyDrive/enhancedlibcem_results/inference',
                dirs_exist_ok=True)
print("✅ Results saved to Drive: /enhancedlibcem_results/")
```

---

## Troubleshooting

### ❌ `No module named 'clip'`
```bash
!pip install -q git+https://github.com/openai/CLIP.git
```

### ❌ `No such file or directory: 'improved_lidecm.pt'`
- Ensure checkpoint is uploaded to `/content/enhancedlibcem/`
- Check: `!ls -la /content/enhancedlibcem/*.pt`

### ❌ CUDA memory issues
```python
import torch
torch.cuda.empty_cache()
```

### ❌ Inference results are black/empty images
- This is expected if checkpoint didn't load correctly
- Verify: `!python quick_test.py` should still work and save diagnostic images

---

## Alternative: Use Our Ready-Made Notebook

Instead of copy-pasting, use **COLAB_FINAL.ipynb**:
1. Download it: `COLAB_FINAL.ipynb`
2. Open in Google Colab: https://colab.research.google.com
3. Upload → Select `COLAB_FINAL.ipynb`
4. Follow the cells step-by-step

---

## Expected Outputs

After running both scripts:

```
/content/enhancedlibcem/quick_test_outputs/
├── original_0.png           (input image)
├── original_1.png
├── reconstructed_0.png      (model reconstruction)
├── reconstructed_1.png
└── diagnostics.txt          (loss, confidence, EM stats)

/content/inference_out/
└── inference_results.png    (5 generated images from text prompts)
```

---

## Performance Notes

- **GPU**: ~10-15 seconds for quick_test + 30 seconds for inference
- **Memory**: ~8-10 GB VRAM (Colab T4 has 16GB, plenty available)
- **Dataset**: CIFAR10 (50K training images, auto-downloaded)

---

**Questions?** Check the main README.md or run:
```bash
!python enhancedlibcem.py --help
!python quick_test.py
!python inference_from_checkpoint.py --help
```
