"""
Cleanup script to remove all generated images and restart training.
This allows for a fresh start without cached data.
"""

import os
import shutil
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def cleanup_generated_files():
    """Remove all generated images, checkpoints, and cached data."""
    
    # Define directories to clean
    cleanup_dirs = [
        'generated_images',
        'output_images',
        'renders',
        'samples',
        'validation_outputs',
        '__pycache__',
        '.cache',
        'checkpoints'
    ]
    
    # Define files to clean
    cleanup_files = [
        'best_model.pt',
        'checkpoint.pt',
        'loss_history.txt',
        'training_log.txt',
        '*.png',
        '*.jpg',
        '*.jpeg'
    ]
    
    logger.info("=" * 60)
    logger.info("CLEANUP: Removing generated files and directories")
    logger.info("=" * 60)
    
    # Remove directories
    for dir_name in cleanup_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                logger.info(f"✓ Removed directory: {dir_name}")
            except Exception as e:
                logger.warning(f"⚠ Could not remove directory {dir_name}: {e}")
        else:
            logger.debug(f"  Directory not found: {dir_name}")
    
    # Remove individual files
    for file_pattern in cleanup_files:
        if '*' in file_pattern:
            # Handle glob patterns
            for file_path in Path('.').glob(file_pattern):
                try:
                    file_path.unlink()
                    logger.info(f"✓ Removed file: {file_path}")
                except Exception as e:
                    logger.warning(f"⚠ Could not remove file {file_path}: {e}")
        else:
            # Handle specific files
            file_path = Path(file_pattern)
            if file_path.exists():
                try:
                    file_path.unlink()
                    logger.info(f"✓ Removed file: {file_pattern}")
                except Exception as e:
                    logger.warning(f"⚠ Could not remove file {file_pattern}: {e}")
            else:
                logger.debug(f"  File not found: {file_pattern}")
    
    logger.info("=" * 60)
    logger.info("✅ CLEANUP COMPLETE - Ready for fresh training!")
    logger.info("=" * 60)

def verify_cleanup():
    """Verify that cleanup was successful."""
    
    logger.info("\nVerifying cleanup...")
    
    check_dirs = [
        'generated_images',
        'output_images',
        'renders',
        'validation_outputs'
    ]
    
    remaining = []
    for dir_name in check_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            remaining.append(dir_name)
    
    if remaining:
        logger.warning(f"⚠ Some directories still exist: {remaining}")
        logger.info("  (This is OK - they can be recreated during training)")
    else:
        logger.info("✓ All output directories removed")
    
    logger.info("\n" + "=" * 60)
    logger.info("Ready to start fresh training!")
    logger.info("=" * 60)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--confirm':
        cleanup_generated_files()
        verify_cleanup()
    else:
        logger.info("\n⚠️  CLEANUP WARNING")
        logger.info("This script will remove:")
        logger.info("  - All generated images and outputs")
        logger.info("  - Model checkpoints")
        logger.info("  - Cache and temporary files")
        logger.info("\nUSAGE:")
        logger.info("  python cleanup.py --confirm")
        logger.info("\nRun with --confirm flag to execute cleanup")
        sys.exit(0)
