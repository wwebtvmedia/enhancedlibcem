# Google Colab Setup - Quick Reference

## Files Created for Colab

1. **COLAB_INSTRUCTIONS.md** - Complete step-by-step guide
2. **COLAB_COPY_PASTE.txt** - Ready-to-copy code for each cell
3. **COLAB_GUIDE.md** - Detailed documentation with troubleshooting
4. **COLAB_READY.py** - Full notebook code with explanations

---

## ⚡ Quick Start (5 minutes)

### Step 1: Go to Google Colab
https://colab.research.google.com

### Step 2: Create New Notebook
Click **"+ New notebook"**

### Step 3: Enable GPU
**Runtime** → **Change runtime type** → Select **GPU** → **Save**

### Step 4: Copy-Paste Setup Code
In **Cell 1**, copy from **COLAB_COPY_PASTE.txt** - **CELL 1: SETUP & INSTALL**

### Step 5: Upload Your Code
1. Click 📁 Files icon
2. Click Upload
3. Select `enhancedlibcem.py`

### Step 6: Copy-Paste Load Code
In **Cell 2**, copy from **COLAB_COPY_PASTE.txt** - **CELL 3: UPLOAD CODE**

### Step 7: Copy-Paste Training or Inference
Choose one:
- **For training**: Copy CELL 5
- **For inference (faster)**: Copy CELL 6

### Step 8: View Results
Copy CELL 7 to display generated images

---

## 📚 Documentation Files

### For Step-by-Step Help
👉 **Read: COLAB_INSTRUCTIONS.md**
- Detailed walkthrough with screenshots
- Troubleshooting section
- Tips & tricks
- Performance benchmarks

### For Copy-Paste Code
👉 **Read: COLAB_COPY_PASTE.txt**
- Ready-to-use code for each cell
- No modifications needed
- Just copy and paste

### For Complete Notebook
👉 **Read: COLAB_READY.py**
- Full notebook with 11 cells
- All functionality in one file
- Comments explaining each step

### For Advanced Usage
👉 **Read: COLAB_GUIDE.md**
- Advanced optimization tips
- Memory management
- GitHub integration
- Detailed explanations

---

## 🎯 Quick Scenarios

### Scenario 1: First Time Training
⏱ Time: ~25 minutes
📋 Steps:
1. Setup & Install (Cell 1)
2. Mount Drive (Cell 2)
3. Upload Code (Cell 3)
4. Train (Cell 5)
5. View Results (Cell 7)

### Scenario 2: Use Pre-trained Model
⏱ Time: ~5 minutes
📋 Steps:
1. Setup & Install (Cell 1)
2. Mount Drive (Cell 2)
3. Load Code (Cell 3)
4. Load Model from Drive (Cell 4)
5. Inference (Cell 6)
6. View Results (Cell 7)

### Scenario 3: Custom Prompts
⏱ Time: ~2 minutes
📋 Steps:
1. Cell 9 from COLAB_COPY_PASTE.txt
2. Edit prompts
3. Run

---

## ✅ Checklist

- [ ] Have Google account
- [ ] Have enhancedlibcem.py ready
- [ ] Go to colab.research.google.com
- [ ] Create new notebook
- [ ] Enable GPU (Runtime → Change runtime type)
- [ ] Copy CELL 1 code and run
- [ ] Upload enhancedlibcem.py
- [ ] Copy CELL 3 code and run
- [ ] Copy training or inference code and run
- [ ] View and download results

---

## 🚀 One-Liner Setup

For advanced users, copy this into ONE cell:

```python
!pip install -q torch torchvision torchaudio git+https://github.com/openai/CLIP.git matplotlib numpy tqdm Pillow; from google.colab import drive; drive.mount('/content/drive'); import os; os.makedirs('/content/data', exist_ok=True); exec(open('/content/enhancedlibcem.py').read()); model, loss = test_tokenization_and_generation('CIFAR10', '/content/data') if not os.path.exists('/content/improved_lidecm.pt') else None; model = run_inference_only('/content/improved_lidecm.pt'); from IPython.display import Image, display; display(Image('/content/inference_results.png')); print("✅ DONE!")
```

---

## 📊 Expected Results

After running, you should see:
- ✅ Training progress bars
- ✅ Loss metrics
- ✅ Generated images from text prompts
- ✅ Model saved to `/content/improved_lidecm.pt`
- ✅ Results saved to Google Drive (if mounted)

---

## 🆘 Common Issues

| Issue | Solution |
|-------|----------|
| GPU not available | Runtime → Change runtime type → Select GPU |
| "No module" errors | Re-run Cell 1 |
| Out of memory | Reduce BATCH_SIZE = 4 |
| Slow training | Check !nvidia-smi (should show 100% GPU) |
| Can't find file | Use Files icon (left) to verify upload |

---

## 📞 Support Resources

- **Read COLAB_INSTRUCTIONS.md** - Troubleshooting section
- **Read COLAB_GUIDE.md** - Advanced help
- **GitHub Issues** - Report bugs
- **Colab Help** - Click ? in Colab

---

## 💾 File Locations in Colab

```
/content/
├── enhancedlibcem.py          ← Upload your code here
├── improved_lidecm.pt         ← Model saves here
├── inference_results.png      ← Generated images
├── data/                      ← Datasets download here
│   └── cifar-10-batches-py/
└── drive/MyDrive/             ← Google Drive (if mounted)
    └── improved_lidecm.pt     ← Backup copy
```

---

## ⚡ Performance

| Task | Time | GPU |
|------|------|-----|
| Setup | 2-3 min | No |
| Download Dataset | 3-5 min | No |
| Training | 12-15 min | Yes (T4/V100) |
| Inference | 2-3 min | Yes |

---

## ✨ Key Features in Colab

✅ Free GPU (NVIDIA T4 or better)
✅ Pre-installed Python & Jupyter
✅ Easy file upload/download
✅ Google Drive integration
✅ Persistent sessions (~12 hours)
✅ 50GB free disk space
✅ Shareable notebooks

---

## 🎓 Learning Path

1. ✅ Run quick training (25 min)
2. ✅ Experiment with different prompts
3. ✅ Save model to Google Drive
4. ✅ Use pre-trained model next session
5. ✅ Fine-tune on custom images
6. ✅ Deploy for production

---

**Ready? Start here: https://colab.research.google.com**

Good luck! 🚀
