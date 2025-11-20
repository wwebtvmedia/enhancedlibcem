# Google Colab Execution Guide
## Enhanced Fractal Tokenization Model

---

## Prerequisites

1. **Google Account** - Required for Google Colab
2. **Google Drive** (optional) - For saving/loading models
3. **GitHub Account** (optional) - For version control

---

## Method 1: Quick Start (Recommended)

### Step 1: Open Google Colab
1. Go to [https://colab.research.google.com](https://colab.research.google.com)
2. Sign in with your Google account
3. Click "New notebook"

### Step 2: Enable GPU Acceleration
1. Go to **Runtime** → **Change runtime type**
2. Select **GPU** as Hardware accelerator
3. Click **Save**

### Step 3: Install Dependencies (Cell 1)
Copy and run this in the first cell:

```python
# Install dependencies
!pip install -q torch torchvision torchaudio
!pip install -q git+https://github.com/openai/CLIP.git
!pip install -q matplotlib numpy tqdm Pillow

# Setup directories
import os
os.makedirs('/content/checkpoints', exist_ok=True)
os.makedirs('/content/data', exist_ok=True)

print("✅ Dependencies installed!")
```

**Expected output**: ✅ Dependencies installed!

### Step 4: Mount Google Drive (Cell 2, Optional)
```python
from google.colab import drive
drive.mount('/content/drive')
print("✓ Google Drive mounted at /content/drive")
```

### Step 5: Upload Code (Cell 3)
**Option A - Direct Upload:**
1. Click the 📁 Files icon on the left
2. Click "Upload" and select `enhancedlibcem.py`
3. Then run:

```python
# Read the uploaded file
with open('/content/enhancedlibcem.py', 'r') as f:
    code = f.read()
exec(code)
```

**Option B - From GitHub:**
```python
!git clone https://github.com/YOUR_USERNAME/enhancedlibcem.git /content/repo
%cd /content/repo
exec(open('enhancedlibcem.py').read())
```

### Step 6: Run Training (Cell 4)
```python
# Train the model
model, loss_history = test_tokenization_and_generation('CIFAR10', '/content/data')
```

**This will:**
- Download CIFAR10 dataset (~170 MB)
- Train for 2 epochs (~10-15 minutes on Colab GPU)
- Save model to `/content/improved_lidecm.pt`
- Generate results: `tokenization_results_*.png`, `generated_images_*.png`

### Step 7: View Results (Cell 5)
```python
from IPython.display import Image, display
from pathlib import Path

# Display all generated images
for img_path in Path('/content').glob('*.png'):
    print(f"\n{img_path.name}")
    display(Image(str(img_path)))
```

---

## Method 2: Using Pre-trained Model (Inference Only)

### If you already have a trained model:

```python
# Cell 1: Setup (same as above)
!pip install -q torch torchvision torchaudio git+https://github.com/openai/CLIP.git

# Cell 2: Load code
exec(open('/content/enhancedlibcem.py').read())

# Cell 3: Load model from Drive (if saved there)
!cp /content/drive/MyDrive/improved_lidecm.pt /content/

# Cell 4: Run inference
custom_prompts = [
    ("a beautiful sunset", 1.0),
    ("abstract art painting", 1.3),
    ("peaceful forest landscape", 0.9),
]

model = run_inference_only('/content/improved_lidecm.pt', custom_prompts)
```

**Time**: ~2 minutes (no training required)

---

## Method 3: Advanced Setup with GitHub

### Clone and Run:

```python
# Cell 1: Clone repository
!git clone https://github.com/YOUR_USERNAME/enhancedlibcem.git /content/repo

# Cell 2: Install dependencies
!pip install -q torch torchvision torchaudio git+https://github.com/openai/CLIP.git

# Cell 3: Run main script
%cd /content/repo
exec(open('enhancedlibcem.py').read())
```

---

## Colab Optimization Tips

### 1. Memory Management
```python
import torch

# Clear GPU cache if you run multiple experiments
torch.cuda.empty_cache()

# Check GPU memory
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### 2. Reduce Memory Usage (if OOM errors)
```python
# Modify constants in enhancedlibcem.py:
BATCH_SIZE = 4  # Reduce from 8
ECM_EPOCHS = 1  # Reduce from 2
NUM_LATENTS = 128  # Reduce from 200
```

### 3. Save to Google Drive
```python
import shutil

# Save model
shutil.copy('/content/improved_lidecm.pt', '/content/drive/MyDrive/')

# Save results
shutil.copy('/content/generated_images_*.png', '/content/drive/MyDrive/')
```

### 4. Monitor Training
```python
# Check GPU during training
!nvidia-smi

# Check available disk space
!df -h /content/
```

---

## Troubleshooting

### Error: "No module named 'clip'"
**Solution:**
```python
!pip install git+https://github.com/openai/CLIP.git
```

### Error: "CUDA out of memory"
**Solution:**
```python
# Reduce batch size
BATCH_SIZE = 4

# Or clear cache
torch.cuda.empty_cache()

# Or use CPU (slower)
DEVICE = 'cpu'
```

### Error: "google.colab not found"
**Solution:** You're not in Google Colab. This is expected if running locally.
```python
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    print("Not in Colab - using local storage")
```

### Error: "The kernel died unexpectedly"
**Solution:** Runtime crashed due to OOM
1. Go to **Runtime** → **Restart runtime**
2. Reduce batch size and epochs
3. Run again

### Slow Training
**Solution:**
1. Check runtime type: **Runtime** → **Change runtime type** → **GPU**
2. Monitor: `!nvidia-smi` (check utilization)
3. If not using GPU, try restarting runtime

---

## Directory Structure in Colab

```
/content/
├── improved_lidecm.pt           # Trained model
├── best_model.pt                # Best checkpoint
├── tokenization_results_*.png   # Tokenization visualizations
├── generated_images_*.png       # Generated results
├── inference_results.png        # Inference outputs
├── training_summary.png         # Loss curves
├── enhancedlibcem.py           # Main code (if uploaded)
├── data/                        # Downloaded datasets
│   └── cifar-10-batches-py/
├── checkpoints/                 # Model checkpoints
└── drive/MyDrive/              # Google Drive (if mounted)
```

---

## Complete Workflow Example

```python
# ===== CELL 1 =====
!pip install -q torch torchvision torchaudio git+https://github.com/openai/CLIP.git
import os
os.makedirs('/content/checkpoints', exist_ok=True)
print("✅ Setup complete")

# ===== CELL 2 =====
from google.colab import drive
drive.mount('/content/drive')

# ===== CELL 3 =====
exec(open('/content/enhancedlibcem.py').read())

# ===== CELL 4 =====
# Check if model exists
import os
if os.path.exists('/content/improved_lidecm.pt'):
    print("Model found - running inference")
    model = run_inference_only('/content/improved_lidecm.pt')
else:
    print("Training new model")
    model, loss = test_tokenization_and_generation('CIFAR10', '/content/data')

# ===== CELL 5 =====
from IPython.display import Image, display
display(Image('/content/inference_results.png'))

# ===== CELL 6 =====
# Save to Drive
import shutil
shutil.copy('/content/improved_lidecm.pt', '/content/drive/MyDrive/')
print("✓ Model saved to Google Drive")
```

---

## Performance Benchmarks (Colab GPU)

| Task | Time | Memory |
|------|------|--------|
| Setup & Install | 2 min | 200 MB |
| Download CIFAR10 | 3 min | 500 MB |
| Training (2 epochs) | 12 min | 4 GB |
| Inference (5 images) | 2 min | 3 GB |
| **Total** | **~20 min** | **~5 GB** |

---

## Next Steps

1. ✅ Train the model
2. ✅ Generate images from text prompts
3. ✅ Experiment with different datasets (CIFAR100, STL10)
4. ✅ Fine-tune on custom images
5. ✅ Deploy for production use

---

## Support & Resources

- **Documentation**: Check COLAB_NOTEBOOK.py for detailed examples
- **Issues**: GitHub Issues
- **Colab Tips**: [Google Colab Guide](https://colab.research.google.com/notebooks/intro.ipynb)

---

**Happy training! 🚀**
