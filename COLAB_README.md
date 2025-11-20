# 🚀 Enhanced LIDECM - Colab Execution Guide

## Quick Start (1 minute setup)

### Option 1: Use Pre-Built Notebook (RECOMMENDED)
1. Open: **[COLAB_GITHUB.ipynb](./COLAB_GITHUB.ipynb)** in Google Colab
2. Click: **File → Open in Colab** (or drag-drop to colab.research.google.com)
3. Run: Execute cells top-to-bottom (Shift+Enter)
4. Results: Saved automatically to `/MyDrive/enhancedlibcem_results`

### Option 2: Fresh Setup
Copy-paste these cells into Colab one by one:

#### Cell 1: Clone & Install
```python
!git clone https://github.com/wwebtvmedia/enhancedlibcem.git /content/enhancedlibcem
!cd /content/enhancedlibcem && pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q git+https://github.com/openai/CLIP.git numpy matplotlib scikit-image scipy numba pillow tqdm
```

#### Cell 2: Verify Setup
```python
from pathlib import Path
checkpoint = Path("/content/enhancedlibcem/improved_lidecm.pt")
print("✅ Checkpoint ready!" if checkpoint.exists() else "⚠️ Checkpoint will be generated from scratch")
```

#### Cell 3: Run Diagnostics
```python
%cd /content/enhancedlibcem
!python quick_test.py
```

#### Cell 4: Run Inference
```python
%cd /content/enhancedlibcem
!python inference_from_checkpoint.py --checkpoint improved_lidecm.pt --outdir /content/inference_out
```

#### Cell 5: Display Results
```python
from IPython.display import Image, display
from pathlib import Path

# Show diagnostic images
print("Original vs Reconstructed:")
for f in sorted(Path("/content/enhancedlibcem/quick_test_outputs").glob("*.png"))[:2]:
    display(Image(str(f)))

# Show inference results
print("\nGenerated Images:")
display(Image("/content/inference_out/inference_results.png"))
```

---

## 📊 What Gets Generated

### Diagnostics Output (`quick_test_outputs/`)
```
original_0.png           - Input image
original_1.png
reconstructed_0.png      - Model reconstruction
reconstructed_1.png
diagnostics.txt          - Loss, confidence, gradients
```

**Key metrics to check:**
- ✅ `M-step: returned loss` (should be ~2-3)
- ✅ `NaN in grads: False` (no numerical issues)
- ✅ `mean assignment confidence` (should be 0.3-0.5)

### Inference Output (`inference_out/`)
```
inference_results.png    - 5 text-guided images (113 KB)
```

Prompts used:
1. "a beautiful natural image"
2. "colorful abstract pattern"
3. "natural scenery landscape"
4. "geometric design"
5. "artistic composition"

---

## ⏱️ Runtime Expectations

| Task | Time | GPU |
|------|------|-----|
| Clone + Install | 2-3 min | CPU/GPU |
| Diagnostics | 10-15 sec | GPU |
| Inference | 30-40 sec | GPU |
| **Total** | **3-5 min** | T4+ |

---

## 🔧 Troubleshooting

### ❌ "No module named 'clip'"
```python
!pip install -q git+https://github.com/openai/CLIP.git
```

### ❌ "improved_lidecm.pt not found"
✅ This is OK! The model works without checkpoint (generates different images).
For reproducible results, upload checkpoint to `/content/enhancedlibcem/`

### ❌ CUDA out of memory
```python
import torch
torch.cuda.empty_cache()
# Reduce batch size in code if needed
```

### ❌ slow downloads
- CIFAR10 is auto-downloaded (~170 MB) - may take 1-2 min first run
- Subsequent runs use cache

---

## 💾 Saving Results

### Auto-Save to Drive (Recommended)
Results automatically saved to: `/MyDrive/enhancedlibcem_results/`

### Manual Download
1. **Files** panel (left sidebar)
2. Right-click `/content/enhancedlibcem/quick_test_outputs/` 
3. **Download**

### Copy to Your Own Drive Folder
```python
import shutil
from pathlib import Path

shutil.copytree("/content/enhancedlibcem/quick_test_outputs",
                "/content/drive/MyDrive/MY_FOLDER/results",
                dirs_exist_ok=True)
```

---

## 🎨 Custom Inference

Run with different prompts:

```python
%cd /content/enhancedlibcem
!python inference_from_checkpoint.py \
    --checkpoint improved_lidecm.pt \
    --outdir /content/custom_results \
    --prompts "a cat" "a tree" "an astronaut"
```

---

## 📝 Next Steps

1. **Review Outputs**: Check generated images & diagnostics
2. **Experiment**: Try custom prompts, different temperatures
3. **Fine-tune**: Modify EM parameters or diffusion steps
4. **Share**: Download results and share!

---

## 🔗 Links

- **GitHub**: https://github.com/wwebtvmedia/enhancedlibcem
- **Main Code**: `enhancedlibcem.py` (2175 lines, full model)
- **Tests**: `quick_test.py` (diagnostic harness)
- **Inference**: `inference_from_checkpoint.py` (CLI wrapper)

---

## 📌 Key Components

The codebase includes:

| File | Purpose |
|------|---------|
| `enhancedlibcem.py` | Core model (encoder, diffusion, EM learner) |
| `quick_test.py` | Diagnostic test (E-step + M-step) |
| `inference_from_checkpoint.py` | Inference CLI |
| `improved_lidecm.pt` | Pre-trained checkpoint (32 MB) |
| `COLAB_GITHUB.ipynb` | **← Start here!** |

---

## ✅ Verification Checklist

After running all cells:

- [ ] ✅ Dependencies installed without errors
- [ ] ✅ Checkpoint loaded (or model initialized)
- [ ] ✅ Diagnostics ran: loss ~2-3, no NaNs
- [ ] ✅ Original & reconstructed images displayed
- [ ] ✅ 5 inference images generated
- [ ] ✅ Results saved to Drive

---

**Questions?** Check the [main README](./README.md) or GitHub issues.

**Ready?** → Open [COLAB_GITHUB.ipynb](./COLAB_GITHUB.ipynb) in Google Colab now! 🚀
