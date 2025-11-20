# 🎨 Enhanced LIDECM - Generative Model with EM Learning

A comprehensive PyTorch implementation of **Enhanced LIDECM** (Latent Iterated Diffusion and EM-based Codebook Modeling) - a state-of-the-art generative model combining:

- 🎯 **Vector Quantized Variational Autoencoders** (VQ-VAE) for discrete representations
- 🌊 **Diffusion Models** for high-quality generation
- 📊 **EM Parameter Learning** with adaptive optimization
- 🔗 **Graph-based Patch Denoising** for spatial coherence
- 💬 **CLIP Integration** for text-guided image generation

---

## 🚀 Quick Start (Colab)

### **One-Click Colab Launch**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wwebtvmedia/enhancedlibcem/blob/main/COLAB_GITHUB.ipynb)

Or read the [**Colab Setup Guide**](./COLAB_README.md)

### **Local Setup (5 minutes)**

```bash
# Clone repository
git clone https://github.com/wwebtvmedia/enhancedlibcem.git
cd enhancedlibcem

# Install dependencies
pip install -r requirements.txt
# or manually:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/openai/CLIP.git numpy matplotlib scikit-image scipy numba pillow tqdm

# Run diagnostics
python quick_test.py

# Run inference
python inference_from_checkpoint.py --checkpoint improved_lidecm.pt
```

---

## 📋 Features

| Feature | Description |
|---------|-------------|
| **E-Step (Tokenization)** | Encode images to discrete token indices using VQ-VAE |
| **M-Step (Optimization)** | Reconstruct images with graph-based patch denoising |
| **EM Learning** | Adaptively learn σ, τ (smoothing), radius parameters |
| **Multi-Head Attention** | Separate attention heads for texture, structure, color, spatial |
| **Diffusion Model** | Generative model in latent space |
| **Text Conditioning** | CLIP embeddings for text-guided generation |
| **Graph Denoising** | Patch-level graph Laplacian smoothing |
| **Perceptual Loss** | VGG16 feature-space reconstruction loss |

---

## 📁 Repository Structure

```
enhancedlibcem/
├── enhancedlibcem.py              # Core model (2175 lines)
├── quick_test.py                  # Diagnostic harness
├── inference_from_checkpoint.py    # Inference CLI
├── colab_inference.py              # Smart Colab loader
├── improved_lidecm.pt              # Pre-trained checkpoint (32 MB)
│
├── COLAB_GITHUB.ipynb             # ⭐ Complete Colab notebook
├── COLAB_README.md                # Full setup guide
├── COLAB_GUIDE_QUICK.md           # Quick reference
├── COLAB_COMPLETE_SUMMARY.md      # Final summary
├── COLAB.md                       # One-click badge
│
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🎯 Usage

### **1. Diagnostic Test**
```bash
python quick_test.py
```
Outputs:
- `quick_test_outputs/original_*.png` - Input images
- `quick_test_outputs/reconstructed_*.png` - Reconstructed images
- `quick_test_outputs/diagnostics.txt` - Loss, gradients, confidence metrics

### **2. Inference from Checkpoint**
```bash
python inference_from_checkpoint.py \
    --checkpoint improved_lidecm.pt \
    --outdir ./results \
    --prompts "a cat" "a blue sky" "geometric pattern"
```
Outputs:
- `inference_results.png` - Grid of 5 generated images (customizable)

### **3. Training (Advanced)**
```python
from enhancedlibcem import EnhancedLIDECM, train_em_step

model = EnhancedLIDECM(dataset_name='CIFAR10')
loss_history = train_em_step(model, epochs=100, learning_rate=1e-3)
model.save_checkpoint('my_checkpoint.pt')
```

---

## 📊 Model Architecture

### **Tokenizer (VQ-VAE)**
- Encoder: Conv layers → continuous latent `z`
- Codebook: `num_tokens` × `latent_dim` learnable vectors
- Decoder: Reconstruct image from quantized codes

### **EM Learner**
- Tracks: `sigma` (spatial scale), `tau` (smoothness), `radius` (neighborhood)
- Per-class parameters for CIFAR10 classes
- Multi-head attention over patch features

### **Diffusion Model**
- Cosine schedule noise addition
- Residual UNet for noise prediction
- Text conditioning via CLIP embeddings

### **Perceptual Loss**
- VGG16 pre-trained on ImageNet
- Feature-space reconstruction loss
- Semantic similarity between original & reconstructed

---

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| Model Parameters | 5.9M |
| Checkpoint Size | 32 MB |
| E-Step Time | ~5 sec (batch_size=2) |
| M-Step Time | ~10 sec (1 optimization step) |
| Inference Time | 30 sec (5 images) |
| VRAM Required | 8-10 GB (T4 GPU) |
| CIFAR10 Accuracy | 0.4-0.5 (reconstruction MSE) |

---

## 🔬 Key Components

### **Graph-Based Patch Denoising**
```python
# Extract patches from images
patches, centers = extract_patches(image, patch_size=8, stride=5)

# Build spatial + feature graph
W = build_patch_graph_radius(centers, patches, radius=10)

# Compute self-similarity
S = compute_self_similarity(patches, W, sigma=0.3)

# Graph Laplacian smoothing
L = graph_laplacian(W)
S_smooth = graph_smoothing(S, L, tau=0.1, steps=5)

# Denoise patches
patches_denoised = denoise_patches_jit(patches, W, S_smooth)

# Reconstruct image
image_reconstructed = reconstruct_image_from_patches(patches_denoised, centers, shape)
```

### **EM Parameter Learning**
```python
# Get current parameters
params = em_learner.get_parameters()
# {'sigma': 0.3, 'tau': 0.1, 'radius': 10.0, ...}

# Compute loss proxy based on reconstruction quality
loss_proxy = em_learner.compute_loss_proxy(recon_loss, percep_loss, smoothness)

# Update parameters with attention weighting
em_learner.update_global_params(loss_proxy, smoothness, patches, centers)

# Get status
status = em_learner.get_status()
# {'global_sigma': '0.3000', 'global_tau': '0.1000', ...}
```

---

## 📦 Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
clip-by-openai (or git+https://github.com/openai/CLIP.git)
numpy>=1.21.0
matplotlib>=3.5.0
scikit-image>=0.19.0
scipy>=1.8.0
numba>=0.56.0
Pillow>=9.0.0
tqdm>=4.60.0
```

See [requirements.txt](./requirements.txt)

---

## 🎓 Educational Value

This implementation demonstrates:

- **VQ-VAE principles** - Discrete latent representations
- **Diffusion models** - Generative modeling via noise scheduling
- **EM algorithm** - Expectation-Maximization for hyperparameter learning
- **Graph signal processing** - Laplacian smoothing, self-similarity
- **CLIP embeddings** - Text-image alignment for conditioning
- **PyTorch best practices** - Modular design, gradient handling, checkpointing
- **Colab integration** - Running complex ML pipelines in the cloud

---

## 🧪 Validation

### **Quick Diagnostic Test**
```bash
python quick_test.py
```
- ✅ Verifies gradient flow (no NaNs/Infs)
- ✅ Checks E-step + M-step execution
- ✅ Saves diagnostic images & metrics
- ✅ Expected loss: ~2-3 (MSE)

### **Inference Verification**
```bash
python inference_from_checkpoint.py
```
- ✅ Loads pre-trained checkpoint
- ✅ Generates 5 images from text prompts
- ✅ Saves composite result image
- ✅ Output: inference_results.png

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: clip` | `pip install git+https://github.com/openai/CLIP.git` |
| CUDA out of memory | Reduce batch size or use CPU (slower) |
| `improved_lidecm.pt` not found | Model runs without checkpoint (generates random) |
| Slow downloads | CIFAR10 (~170 MB) cached after first run |
| NaN loss values | Reduce learning rate or gradient clip norm |

See [COLAB_README.md](./COLAB_README.md) for more troubleshooting.

---

## 📚 References

- **VQ-VAE**: [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)
- **Diffusion**: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- **CLIP**: [Learning Transferable Models for Computational Imaging](https://arxiv.org/abs/2103.14030)
- **EM Algorithm**: [Expectation–Maximization Algorithm](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm)
- **Graph Signal Processing**: [Graph Signal Processing](https://arxiv.org/abs/1211.0053)

---

## 🎨 Example Outputs

### Diagnostic Test
- **Input**: Random CIFAR10-like images (32×32)
- **Reconstructed**: Model reconstruction (preserves structure)
- **Metrics**: M-step loss ~2.68, no gradient issues

### Inference Results
5 text-guided images generated from prompts:
1. "a beautiful natural image"
2. "colorful abstract pattern"
3. "natural scenery landscape"
4. "geometric design"
5. "artistic composition"

---

## 💾 Checkpoint Information

**improved_lidecm.pt** (32 MB):
- Trained on CIFAR10 dataset
- Contains: tokenizer, generator, renderer, diffusion, optimizer states
- Compatible with all inference scripts
- Auto-downloaded or can be loaded from Drive in Colab

---

## 🚀 Next Steps

1. **Try Colab**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wwebtvmedia/enhancedlibcem/blob/main/COLAB_GITHUB.ipynb)
2. **Run locally**: `python quick_test.py` then `python inference_from_checkpoint.py`
3. **Experiment**: Modify prompts, EM parameters, diffusion steps
4. **Fine-tune**: Train on custom datasets
5. **Share**: Upload results to social media! 📸

---

## 📝 License

[Add your license here - e.g., MIT, Apache 2.0, etc.]

---

## 👥 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📞 Support

- **Colab Issues**: See [COLAB_README.md](./COLAB_README.md)
- **Code Issues**: Open a GitHub issue
- **Questions**: Check [COLAB_COMPLETE_SUMMARY.md](./COLAB_COMPLETE_SUMMARY.md)

---

## 🎉 Acknowledgments

- CIFAR10 dataset (learning resource)
- PyTorch team (excellent deep learning framework)
- OpenAI CLIP (text-image alignment)
- All contributors and users!

---

**Happy generating! 🎨**

*For quick start, click the Colab badge above ↑*
