"""
colab_inference.py

Inference script optimized for Colab environment.
Handles checkpoint from Drive and generates images.

Usage:
    python colab_inference.py --checkpoint-path /path/to/improved_lidecm.pt
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('colab_inference')

def find_checkpoint(hint_paths=None):
    """Find checkpoint in common Colab locations"""
    if hint_paths is None:
        hint_paths = [
            './improved_lidecm.pt',
            '/content/improved_lidecm.pt',
            '/content/enhancedlibcem/improved_lidecm.pt',
            '/content/drive/MyDrive/improved_lidecm.pt',
        ]
    
    for path in hint_paths:
        p = Path(path)
        if p.exists():
            logger.info(f"✓ Found checkpoint: {path}")
            return str(p)
    
    logger.warning("⚠ Checkpoint not found in common locations")
    return None

def run_inference(checkpoint_path=None, outdir='./inference_out'):
    """Run inference with fallback handling"""
    from pathlib import Path
    import torch
    
    # Create output directory
    Path(outdir).mkdir(exist_ok=True)
    
    # Find checkpoint if not provided
    if checkpoint_path is None:
        checkpoint_path = find_checkpoint()
    
    if checkpoint_path is None:
        logger.error("❌ Checkpoint not found. Please provide --checkpoint-path or upload improved_lidecm.pt")
        logger.info("\nAlternative: Run quick_test.py first to verify model works without checkpoint:")
        logger.info("  python quick_test.py")
        return False
    
    try:
        from enhancedlibcem import run_inference_only
        
        logger.info(f"\n{'='*80}")
        logger.info("COLAB INFERENCE - LOADING PRE-TRAINED MODEL")
        logger.info(f"{'='*80}\n")
        
        # Run inference
        model = run_inference_only(checkpoint_path=checkpoint_path)
        
        if model is not None:
            # Copy results to output directory
            src = Path('inference_results.png')
            if src.exists():
                import shutil
                dst = Path(outdir) / 'inference_results.png'
                shutil.copy(src, dst)
                logger.info(f"✓ Results copied to: {dst}")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on Colab')
    parser.add_argument('--checkpoint-path', type=str, default=None,
                       help='Path to checkpoint file (default: auto-search)')
    parser.add_argument('--outdir', type=str, default='./inference_out',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    success = run_inference(checkpoint_path=args.checkpoint_path, outdir=args.outdir)
    
    if success:
        logger.info("\n✅ Inference completed successfully!")
        sys.exit(0)
    else:
        logger.info("\n⚠ Inference encountered issues. Check logs above.")
        sys.exit(1)
