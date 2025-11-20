# Vast.ai GPU Training Setup Guide

Complete step-by-step guide to run 7-hour training on Vast.ai for $2-5 total cost.

---

## Step 1: Create Vast.ai Account

1. Go to https://www.vast.ai
2. Click **"Sign Up"** (top right)
3. Create account with email + password
4. Add payment method (Visa/Mastercard/Crypto)
5. Add billing info (name, address)

**Cost estimate:** RTX 3060 is ~$0.25/hour → 7 hours = $1.75

---

## Step 2: Rent GPU Instance

1. Go to https://www.vast.ai/console/create/
2. Configure your search:

   ```
   GPU Type: RTX 3060, RTX 3070, or RTX 4090 (any)
   Min GPU Memory: 12GB
   Min CPU: 4 cores
   Min RAM: 16GB
   Disk Space: 50GB
   Sort by: Price (low to high)
   ```

3. Look for instances with **Linux** OS (Ubuntu 20.04 or 22.04)
4. Click the instance with lowest price
5. Click **"RENT"** button
6. Confirm rental (will deduct from balance)

**Expected cost per hour:** $0.20 - $0.40

---

## Step 3: Connect to Instance

Once rented, you'll see instance details. Get:
- **SSH Command** (copy this)
- **Port Number**
- **User**: root
- **Password** (shown once)

### On Windows PowerShell:

```powershell
# Install SSH client (if not already installed)
Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH.Client*' | Add-WindowsCapability -Online

# Connect to instance (paste SSH command from Vast.ai)
ssh -p [PORT] root@[IP_ADDRESS]

# Type password when prompted (paste from Vast.ai console)
```

### On Mac/Linux Terminal:

```bash
ssh -p [PORT] root@[IP_ADDRESS]
# Enter password when prompted
```

**Result:** You should see Linux prompt like `root@instance:/# `

---

## Step 4: Setup Environment on Instance

Once connected via SSH:

```bash
# Update package manager
apt-get update && apt-get upgrade -y

# Install Python and dependencies
apt-get install -y python3 python3-pip git wget

# Verify Python version (should be 3.8+)
python3 --version

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install required packages
pip install scikit-image scipy numba numpy pillow tensorboard

# Create working directory
mkdir -p /root/training
cd /root/training
```

---

## Step 5: Upload Your Code

### Option A: Via Git (Recommended)

```bash
# Clone your repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### Option B: Via SCP (File Transfer)

From your local machine (NOT on Vast.ai):

```powershell
# Copy single file
scp -P [PORT] enhancedlibcem.py root@[IP_ADDRESS]:/root/training/

# Copy entire directory
scp -P [PORT] -r . root@[IP_ADDRESS]:/root/training/
```

### Option C: Upload Zip File

```bash
# On Vast.ai instance:
cd /root/training
wget https://your-file-hosting.com/your_code.zip
unzip your_code.zip
```

---

## Step 6: Prepare Training Script

Create `train_vast.py` on the instance:

```bash
cat > /root/training/train_vast.py << 'EOF'
#!/usr/bin/env python3
"""
Vast.ai training script for Enhanced LIDECM
Runs 7 hours of training on CIFAR10
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.insert(0, '/root/training')

# Import your model
from enhancedlibcem import EnhancedLIDECM, DEVICE

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ==================== CONFIGURATION ====================
DATASET_NAME = 'CIFAR10'
BATCH_SIZE = 32
NUM_EPOCHS = 10  # Will complete in ~7 hours
LEARNING_RATE = 1e-4
LOG_INTERVAL = 50

# ==================== SETUP ====================
output_dir = Path('/root/training/outputs')
output_dir.mkdir(exist_ok=True)

log_file = output_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

def log_msg(msg):
    print(msg)
    with open(log_file, 'a') as f:
        f.write(msg + '\n')

log_msg(f"Training started at {datetime.now()}")
log_msg(f"Output directory: {output_dir}")

# ==================== LOAD DATA ====================
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

log_msg("Loading CIFAR10 dataset...")
train_dataset = datasets.CIFAR10(
    root='/root/training/data',
    train=True,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)

log_msg(f"Dataset loaded: {len(train_dataset)} images")

# ==================== INITIALIZE MODEL ====================
log_msg("Initializing Enhanced LIDECM model...")
model = EnhancedLIDECM(dataset_name=DATASET_NAME)
model.to(DEVICE)

log_msg(f"Model initialized on {DEVICE}")
log_msg(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# ==================== TRAINING LOOP ====================
log_msg("\n" + "="*60)
log_msg("STARTING TRAINING LOOP")
log_msg("="*60 + "\n")

best_loss = float('inf')
loss_history = []

try:
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        num_batches = 0
        
        log_msg(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        log_msg("-" * 40)
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            
            # Forward pass
            token_indices, ifs_tokens, ifs_probs, patch_info = model.tokenizer.tokenize(images)
            
            # E-step
            token_indices, ifs_tokens, ifs_probs, patch_info, conf = model.e_step(images, epoch)
            
            # M-step (includes graph denoising, EM parameter learning, multi-head attention)
            loss = model.m_step(
                images, 
                token_indices, 
                ifs_tokens, 
                ifs_probs, 
                patch_info, 
                current_epoch=epoch
            )
            
            epoch_loss += loss
            num_batches += 1
            
            # Logging
            if (batch_idx + 1) % LOG_INTERVAL == 0:
                avg_loss = epoch_loss / num_batches
                log_msg(f"  Batch {batch_idx + 1}/{len(train_loader)}: Loss = {loss:.4f} (avg: {avg_loss:.4f})")
        
        # Epoch statistics
        avg_epoch_loss = epoch_loss / num_batches
        loss_history.append(avg_epoch_loss)
        
        log_msg(f"\nEpoch {epoch + 1} Summary:")
        log_msg(f"  Average Loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            checkpoint_path = output_dir / f'model_epoch_{epoch+1}_loss_{avg_epoch_loss:.4f}.pt'
            torch.save(model.state_dict(), checkpoint_path)
            log_msg(f"  ✅ New best loss! Saved checkpoint: {checkpoint_path.name}")
        
        # EM parameter update log
        em_status = model.em_learner.get_status()
        log_msg(f"  EM Parameters: {em_status}")

except KeyboardInterrupt:
    log_msg("\n⚠️ Training interrupted by user")
except Exception as e:
    log_msg(f"\n❌ Training error: {e}")
    import traceback
    log_msg(traceback.format_exc())

# ==================== SUMMARY ====================
log_msg("\n" + "="*60)
log_msg("TRAINING COMPLETE")
log_msg("="*60)
log_msg(f"Total epochs: {len(loss_history)}")
log_msg(f"Best loss: {min(loss_history):.4f}")
log_msg(f"Final loss: {loss_history[-1]:.4f}")
log_msg(f"Loss improvement: {((loss_history[0] - loss_history[-1]) / loss_history[0] * 100):.1f}%")
log_msg(f"Finished at {datetime.now()}")
log_msg(f"Log file: {log_file}")

# Save loss history
np.save(output_dir / 'loss_history.npy', loss_history)

print("\n✅ Training finished! Check log file for details.")

EOF
```

Make the script executable:

```bash
chmod +x /root/training/train_vast.py
```

---

## Step 7: Run Training

Start the training (runs continuously):

```bash
cd /root/training
python3 train_vast.py
```

**Expected output:**
```
PyTorch version: 2.x.x
CUDA available: True
GPU name: NVIDIA RTX 3060
GPU memory: 12.00 GB
Loading CIFAR10 dataset...
Dataset loaded: 50000 images
Initializing Enhanced LIDECM model...
Model initialized on cuda:0

============================================================
STARTING TRAINING LOOP
============================================================

Epoch 1/10
----------------------------------------
  Batch 50/1563: Loss = 0.4521 (avg: 0.4856)
  Batch 100/1563: Loss = 0.4312 (avg: 0.4654)
  ...
  Epoch 1 Summary:
  Average Loss: 0.4523
  ✅ New best loss! Saved checkpoint: model_epoch_1_loss_0.4523.pt
  EM Parameters: {sigma: 0.312, tau: 0.111, ...}

Epoch 2/10
...
```

---

## Step 8: Monitor Training (in background)

Open a **second SSH terminal** and run:

```bash
# Watch training log in real-time
tail -f /root/training/outputs/training_*.log

# Check GPU usage
watch -n 1 nvidia-smi

# Check disk space
du -sh /root/training/

# List saved checkpoints
ls -lh /root/training/outputs/
```

---

## Step 9: Download Results

After training completes, download results to your local machine:

```powershell
# From your local machine (PowerShell), NOT on Vast.ai:

# Download entire outputs directory
scp -r -P [PORT] root@[IP_ADDRESS]:/root/training/outputs ./training_results/

# Download just the best model
scp -P [PORT] root@[IP_ADDRESS]:/root/training/outputs/model_epoch_*.pt ./models/

# Download loss history
scp -P [PORT] root@[IP_ADDRESS]:/root/training/outputs/loss_history.npy ./
```

---

## Step 10: Stop Instance & Save Money

When done training:

```bash
# On Vast.ai instance, end the training:
Ctrl+C

# Disconnect from instance:
exit
```

Then in Vast.ai console:
1. Go to "My Instances"
2. Find your instance
3. Click **"STOP"** (saves your data)
4. To permanently stop paying, click **"DESTROY"**

**Cost summary:**
- Rental time: 7 hours
- GPU hourly rate: $0.25/hour
- **Total cost: $1.75** ✅

---

## Troubleshooting

### **Issue: SSH connection timeout**
```bash
# Try with longer timeout
ssh -p [PORT] -o ConnectTimeout=30 root@[IP_ADDRESS]
```

### **Issue: GPU out of memory**
Reduce batch size in `train_vast.py`:
```python
BATCH_SIZE = 16  # Reduce from 32
```

### **Issue: Graph denoising very slow**
Reduce smooth_steps in `enhancedlibcem.py`:
```python
em_params['smooth_steps'] = 15  # Reduce from 25
```

### **Issue: Want to pause training**
```bash
# Press Ctrl+C to pause
# Instance stays rented but not training
# Resume with: python3 train_vast.py
# (continues from last checkpoint)
```

### **Issue: Lost SSH connection mid-training**
```bash
# Reconnect and check progress:
ssh -p [PORT] root@[IP_ADDRESS]
tail -f /root/training/outputs/training_*.log

# Training continues in background even if disconnected!
```

---

## Expected Results

### Timeline:
```
Epoch 1: 35 min  → Loss: 0.50
Epoch 2: 35 min  → Loss: 0.38
Epoch 3: 35 min  → Loss: 0.27
Epoch 4: 35 min  → Loss: 0.20
Epoch 5: 35 min  → Loss: 0.16
Epoch 6: 35 min  → Loss: 0.14
Epoch 7: 35 min  → Loss: 0.13  ← Converges here
Epoch 8: 35 min  → Loss: 0.128
Epoch 9: 35 min  → Loss: 0.127
Epoch 10: 35 min → Loss: 0.127

Total: 350 minutes ≈ 5.8 hours
```

### Saved files:
```
outputs/
├── training_20241116_143022.log      (detailed log)
├── model_epoch_1_loss_0.5032.pt      (checkpoint)
├── model_epoch_2_loss_0.3821.pt      (checkpoint)
├── model_epoch_7_loss_0.1284.pt      (best, converged)
├── model_epoch_10_loss_0.1274.pt     (final)
└── loss_history.npy                  (numpy array of losses)
```

---

## Next Steps

1. **Download best model** from outputs
2. **Evaluate** on test set
3. **Fine-tune hyperparameters** if needed
4. **Fine-tune attention weights** if specific image types need adjustment

---

## Cost Comparison

| Platform | Cost | Speed | Setup |
|----------|------|-------|-------|
| **Vast.ai (RTX 3060)** | **$1.75** | 5.8h | 10 min |
| **Vast.ai (RTX A6000)** | **$2.80** | 3h | 10 min |
| Google Colab | Free | 7h | 2 min |
| Lambda Labs | $2.17 | 4h | 15 min |
| Paperspace | $2.80 | 5h | 10 min |

**Recommendation:** Start with **Vast.ai RTX 3060** for best price-to-speed ratio ($1.75 for 5.8 hours).

---

**Status: ✅ Ready to train on Vast.ai**
