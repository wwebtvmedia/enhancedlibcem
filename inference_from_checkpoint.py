"""
inference_from_checkpoint.py

Loads a pre-trained checkpoint (default: improved_lidecm.pt) and runs
text-guided inference using `run_inference_only` defined in `enhancedlibcem.py`.

Usage examples (PowerShell):
    # Run with default checkpoint in workspace
    python .\inference_from_checkpoint.py

    # Specify checkpoint path and output directory
    python .\inference_from_checkpoint.py --checkpoint .\improved_lidecm.pt --outdir .\inference_out

In Colab:
    Upload `improved_lidecm.pt` to `/content/` and run:
    !python /content/inference_from_checkpoint.py --checkpoint /content/improved_lidecm.pt --outdir /content/outputs

The script will call `run_inference_only(checkpoint_path, prompts)` inside `enhancedlibcem.py`.
"""

import argparse
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', '-c', type=str, default='improved_lidecm.pt', help='Path to checkpoint file')
parser.add_argument('--outdir', '-o', type=str, default='inference_outputs', help='Output directory')
parser.add_argument('--prompts', '-p', type=str, nargs='*', help='Optional prompts (wrap multi-word prompts in quotes)')
args = parser.parse_args()

OUTDIR = Path(args.outdir)
OUTDIR.mkdir(parents=True, exist_ok=True)

# Ensure repo path is importable
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from enhancedlibcem import run_inference_only
except Exception as e:
    logger.error(f"Could not import run_inference_only from enhancedlibcem: {e}")
    raise

# Prepare prompts if given, else let function use defaults
prompts = None
if args.prompts and len(args.prompts) > 0:
    # Interpret prompts as prompt|temp pairs if provided in that format, otherwise use default temp=1.0
    parsed = []
    for p in args.prompts:
        if '|' in p:
            txt, t = p.split('|', 1)
            try:
                temp = float(t)
            except Exception:
                temp = 1.0
            parsed.append((txt, temp))
        else:
            parsed.append((p, 1.0))
    prompts = parsed

logger.info(f"Running inference with checkpoint: {args.checkpoint}")
model = run_inference_only(checkpoint_path=args.checkpoint, prompts=prompts)

# Copy outputs (inference_results.png expected)
generated = Path('inference_results.png')
if generated.exists():
    dest = OUTDIR / generated.name
    with open(generated, 'rb') as rf, open(dest, 'wb') as wf:
        wf.write(rf.read())
    logger.info(f"Saved inference image to: {dest}")
else:
    logger.warning("inference_results.png not found. Check logs for errors.")

logger.info(f"Inference outputs are in: {OUTDIR.resolve()}")
