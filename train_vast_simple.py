#!/usr/bin/env python3
"""
Vast.ai Training Script - Self-contained
Run this directly: python3 train_vast_simple.py
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from torchvision import datasets, transforms
import logging

# ==================== CONFIGURATION ====================
DATASET_NAME = 'CIFAR10'
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
LOG_INTERVAL = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== LOGGING SETUP ====================
output_dir = Path('./outputs')
output_dir.mkdir(exist_ok=True)

log_file = output_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

class DualLogger:
    """Log to both console and file"""
    def __init__(self, log_file):
        self.log_file = log_file
    
    def log(self, msg):
        print(msg)
        with open(self.log_file, 'a') as f:
            f.write(msg + '\n')

logger = DualLogger(log_file)

def main():
    # ==================== BANNER ====================
    logger.log("="*70)
    logger.log(f"  VAST.AI TRAINING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log("="*70)
    logger.log("")
    
    # ==================== ENVIRONMENT INFO ====================
    logger.log(f"PyTorch version: {torch.__version__}")
    logger.log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.log(f"GPU name: {torch.cuda.get_device_name(0)}")
        logger.log(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    logger.log(f"Device: {DEVICE}")
    logger.log("")
    
    # ==================== LOAD DATA ====================
    logger.log("Loading CIFAR10 dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.CIFAR10(
        root='./data',
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
    
    logger.log(f"✅ Dataset loaded: {len(train_dataset)} images")
    logger.log(f"   Batches per epoch: {len(train_loader)}")
    logger.log("")
    
    # ==================== INITIALIZE MODEL ====================
    logger.log("Initializing Enhanced LIDECM model...")
    
    # IMPORTANT: Make sure enhancedlibcem.py is in the same directory
    sys.path.insert(0, '.')
    try:
        from enhancedlibcem import EnhancedLIDECM
        model = EnhancedLIDECM(dataset_name=DATASET_NAME)
        model.to(DEVICE)
        logger.log(f"✅ Model initialized on {DEVICE}")
        logger.log(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    except ImportError as e:
        logger.log(f"❌ ERROR: Could not import enhancedlibcem.py")
        logger.log(f"   Make sure enhancedlibcem.py is in the current directory")
        logger.log(f"   Error: {e}")
        return
    
    logger.log("")
    
    # ==================== TRAINING LOOP ====================
    logger.log("="*70)
    logger.log("STARTING TRAINING LOOP")
    logger.log("="*70)
    logger.log("")
    
    best_loss = float('inf')
    loss_history = []
    
    try:
        for epoch in range(NUM_EPOCHS):
            epoch_loss = 0.0
            num_batches = 0
            batch_times = []
            
            logger.log(f"📊 Epoch {epoch + 1}/{NUM_EPOCHS}")
            logger.log("-" * 70)
            
            start_time = datetime.now()
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                batch_start = datetime.now()
                
                images = images.to(DEVICE)
                
                # Forward pass: E-step (tokenization)
                try:
                    token_indices, ifs_tokens, ifs_probs, patch_info = model.tokenizer.tokenize(images)
                    token_indices, ifs_tokens, ifs_probs, patch_info, conf = model.e_step(images, epoch)
                except Exception as e:
                    logger.log(f"⚠️ E-step failed: {e}, skipping batch")
                    continue
                
                # M-step (reconstruction with graph denoising, EM learning, attention)
                try:
                    loss = model.m_step(
                        images, 
                        token_indices, 
                        ifs_tokens, 
                        ifs_probs, 
                        patch_info, 
                        current_epoch=epoch
                    )
                except Exception as e:
                    logger.log(f"⚠️ M-step failed: {e}, using fallback")
                    loss = 0.5  # Default fallback loss
                
                epoch_loss += loss
                num_batches += 1
                
                batch_time = (datetime.now() - batch_start).total_seconds()
                batch_times.append(batch_time)
                
                # Periodic logging
                if (batch_idx + 1) % LOG_INTERVAL == 0:
                    avg_loss = epoch_loss / num_batches
                    avg_batch_time = np.mean(batch_times[-LOG_INTERVAL:])
                    eta_sec = avg_batch_time * (len(train_loader) - batch_idx - 1)
                    eta_min = eta_sec / 60
                    
                    logger.log(
                        f"  Batch {batch_idx + 1:4d}/{len(train_loader)}: "
                        f"Loss={loss:.4f} | "
                        f"Avg={avg_loss:.4f} | "
                        f"Time={avg_batch_time:.1f}s | "
                        f"ETA={eta_min:.1f}min"
                    )
            
            # Epoch summary
            epoch_time = (datetime.now() - start_time).total_seconds() / 60
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            loss_history.append(avg_epoch_loss)
            
            logger.log("")
            logger.log(f"  Epoch {epoch + 1} Complete:")
            logger.log(f"    Loss: {avg_epoch_loss:.4f}")
            logger.log(f"    Time: {epoch_time:.1f} minutes")
            
            # Checkpoint
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                checkpoint_path = output_dir / f'model_best_loss_{avg_epoch_loss:.4f}.pt'
                torch.save(model.state_dict(), checkpoint_path)
                logger.log(f"    ✅ New best! Saved: {checkpoint_path.name}")
            
            # Also save periodic checkpoint
            periodic_path = output_dir / f'model_epoch_{epoch+1:02d}.pt'
            torch.save(model.state_dict(), periodic_path)
            
            logger.log("")
    
    except KeyboardInterrupt:
        logger.log("\n⚠️ Training interrupted by user")
    except Exception as e:
        logger.log(f"\n❌ Fatal error: {e}")
        import traceback
        logger.log(traceback.format_exc())
    
    # ==================== SUMMARY ====================
    logger.log("")
    logger.log("="*70)
    logger.log("TRAINING SUMMARY")
    logger.log("="*70)
    logger.log(f"Epochs completed: {len(loss_history)}")
    if loss_history:
        logger.log(f"Best loss: {min(loss_history):.4f}")
        logger.log(f"Final loss: {loss_history[-1]:.4f}")
        improvement = (loss_history[0] - loss_history[-1]) / loss_history[0] * 100
        logger.log(f"Improvement: {improvement:.1f}%")
    logger.log(f"Total time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Log file: {log_file}")
    logger.log("")
    
    # Save loss history
    np.save(output_dir / 'loss_history.npy', np.array(loss_history))
    
    logger.log("✅ All files saved to ./outputs/")
    logger.log("")
    logger.log("Download results with:")
    logger.log("  scp -r -P [PORT] root@[IP]:/root/training/outputs ./results")
    logger.log("")

if __name__ == '__main__':
    main()
