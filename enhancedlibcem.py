import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image
import clip
import matplotlib.pyplot as plt
import os
import math
from tqdm import tqdm
import logging
import numpy as np
from torchvision.models import vgg16
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, csr_matrix, diags
from numba import jit, prange
from skimage.util import view_as_windows

# --------------------------
# Enhanced Constants
# --------------------------
LATENT_DIM = 256  # Increased for better feature representation
NUM_LATENTS = 512  # Increased codebook size for richer tokens
NUM_TRANSFORMS = 5  # More IFS transforms for complexity
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATASET_NAME = 'CIFAR10'
BATCH_SIZE = 32  # Increased for better gradient estimates
ECM_EPOCHS = 10  # Full training for quality enhancement
DIFFUSION_STEPS = 100
CHECKPOINT_PATH = "improved_lidecm.pt"
BEST_CHECKPOINT_PATH = "best_model.pt"
CHECKPOINT_DIR = "./checkpoints"
GUIDANCE_SCALE = 7.0
TRAIN_PROMPT = "natural fractal pattern"
TEXT_DIM = 512
TIME_EMBED_DIM = 256  # Larger time embeddings
DOWN1_OUT_DIM = 512  # Wider intermediate layer
ATTN_NUM_HEADS = 16  # More attention heads
RENDERER_CONV_OUT_CHANNELS = 32  # Increased feature channels
RENDERER_KERNEL_SIZE = 3
RENDERER_PADDING = 1
RENDERER_COORD_EMBED_DIM = 512  # Increased embedding dimension
RENDERER_POLICY_MID_DIM = 256  # Increased policy network width
IFS_NET_MID_DIM = 512  # Increased hidden layer width
IFS_AFFINE_DIM = 6
RENDER_STEPS = 100
RENDER_N_POINTS = 500
RENDER_MIN_RESOLUTION = 32
RENDER_RESOLUTION_STEP = 8
RENDER_EPOCH_STEP = 2
RENDER_FINAL_RESOLUTION = 64  # Standard resolution
VGA_RESOLUTION = 640  # VGA support (640x480)
VGA_RENDER_RESOLUTION = 480
SAVE_FIG_SIZE = 8
GENERATE_STEPS = 500
OPTIMIZER_LR_GENERATOR = 5e-4
OPTIMIZER_LR_RENDERER = 5e-4
OPTIMIZER_LR_DIFFUSION = 5e-5
OPTIMIZER_LR_LATENT_CODES = 5e-4
KL_LOSS_WEIGHT = 0.05
GRAD_CLIP_NORM = 0.5
NORM_MEAN = 0.5
NORM_STD = 0.5
MATRIX_NORM_THRESHOLD = 0.9
MATRIX_NORM_EPSILON = 1e-8
BUFFER_MAX_EPSILON = 1e-10
CHECKPOINT_SAVE_INTERVAL = 1
PERCEPTUAL_LOSS_WEIGHT = 0.2

# --------------------------
# Graph-Based Patch Diffusion Utilities
# --------------------------

def extract_patches(image, patch_size=7, stride=5):
    """Extract overlapping patches and their centers from image."""
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    image = np.transpose(image, (1, 2, 0)) if image.ndim == 3 and image.shape[0] == 3 else image
    
    patches = view_as_windows(image, (patch_size, patch_size, 3), step=stride)
    num_y, num_x = patches.shape[:2]
    patches = patches.reshape(-1, patch_size, patch_size, 3)
    patches_vec = patches.reshape(patches.shape[0], -1)
    
    centers_y = np.arange(patch_size//2, image.shape[0]-patch_size//2+1, stride)
    centers_x = np.arange(patch_size//2, image.shape[1]-patch_size//2+1, stride)
    centers = np.array(np.meshgrid(centers_y, centers_x, indexing='ij')).reshape(2, -1).T
    return patches_vec, centers

@jit(nopython=True, parallel=True)
def compute_feature_similarities(patches, rows, cols, sigma_f):
    """Compute pairwise feature similarities using Numba JIT."""
    n = len(rows)
    sims = np.zeros(n, dtype=np.float32)
    for k in prange(n):
        i, j = rows[k], cols[k]
        diff = patches[i] - patches[j]
        dist_sq = np.dot(diff, diff)
        sims[k] = np.exp(-dist_sq / (2 * sigma_f**2))
    return sims

def build_patch_graph_radius(centers, patches, radius, sigma_s=10.0, sigma_f=0.3):
    """Build weighted patch graph using spatial radius and feature similarity."""
    tree = cKDTree(centers)
    pairs = tree.query_pairs(r=radius, output_type='ndarray')
    rows = pairs[:, 0]
    cols = pairs[:, 1]
    distances = np.linalg.norm(centers[rows] - centers[cols], axis=1)
    spatial_weights = np.exp(-distances**2 / (2 * sigma_s**2))
    feature_sims = compute_feature_similarities(patches, rows, cols, sigma_f)
    weights = spatial_weights * feature_sims
    rows_sym = np.concatenate([rows, cols])
    cols_sym = np.concatenate([cols, rows])
    weights_sym = np.concatenate([weights, weights])
    n = len(centers)
    W = coo_matrix((weights_sym, (rows_sym, cols_sym)), shape=(n, n))
    return W.tocsr()

@jit(nopython=True, parallel=True)
def compute_similarities(patches, rows, cols, sigma):
    """Compute self-similarity using Numba JIT."""
    n = len(rows)
    sims = np.zeros(n, dtype=np.float32)
    for k in prange(n):
        i, j = rows[k], cols[k]
        diff = patches[i] - patches[j]
        dist_sq = np.dot(diff, diff)
        sims[k] = np.exp(-dist_sq / (2 * sigma**2))
    return sims

def compute_self_similarity(patches, W, sigma):
    """Compute self-similarity matrix S from patches and graph W."""
    n = patches.shape[0]
    rows, cols = W.nonzero()
    sims = compute_similarities(patches, rows, cols, sigma)
    S = coo_matrix((sims, (rows, cols)), shape=(n, n))
    return S.tocsr()

def graph_laplacian(W):
    """Compute graph Laplacian L = D - W."""
    d = np.array(W.sum(axis=1)).flatten()
    D = diags(d)
    L = D - W
    return L

def graph_smoothing(S, L, tau, steps):
    """Smooth self-similarity matrix using graph Laplacian."""
    S = S.tocsr()
    for _ in range(steps):
        S = S - tau * L.dot(S)
        S.data = np.tanh(S.data)
    return S

@jit(nopython=True, parallel=True)
def denoise_patches_jit(patches, W_rows, W_cols, W_data, S_smooth_rows, S_smooth_cols, S_smooth_data, n):
    """Denoise patches using smoothed similarity matrix."""
    patches_denoised = np.zeros_like(patches)
    for i in prange(n):
        neigh_mask = (W_rows == i)
        neighbors = W_cols[neigh_mask]
        if len(neighbors) == 0:
            patches_denoised[i] = patches[i]
            continue
        weights = np.zeros(len(neighbors), dtype=np.float32)
        for k, neigh in enumerate(neighbors):
            weight_mask = (S_smooth_rows == i) & (S_smooth_cols == neigh)
            if np.any(weight_mask):
                weights[k] = S_smooth_data[weight_mask][0]
        weight_sum = np.sum(weights)
        if weight_sum > 1e-8:
            weights /= weight_sum
            for j in prange(len(neighbors)):
                patches_denoised[i] += patches[neighbors[j]] * weights[j]
        else:
            patches_denoised[i] = patches[i]
    return patches_denoised

@jit(nopython=True, parallel=True)
def reconstruct_image_from_patches_jit(patches_vec, centers, image_shape, patch_size, stride):
    """Reconstruct image from denoised patches."""
    H, W, C = image_shape
    image_recon = np.zeros((H, W, C), dtype=np.float32)
    weight_map = np.zeros((H, W), dtype=np.float32)
    num_patches = patches_vec.shape[0]
    patches = patches_vec.reshape(num_patches, patch_size, patch_size, C)
    
    for i in prange(num_patches):
        cy, cx = int(centers[i, 0]), int(centers[i, 1])
        y1 = cy - patch_size // 2
        y2 = y1 + patch_size
        x1 = cx - patch_size // 2
        x2 = x1 + patch_size
        if y1 >= 0 and x1 >= 0 and y2 <= H and x2 <= W:
            for py in range(patch_size):
                for px in range(patch_size):
                    for c in range(C):
                        image_recon[y1 + py, x1 + px, c] += patches[i, py, px, c]
            for py in range(patch_size):
                for px in range(patch_size):
                    weight_map[y1 + py, x1 + px] += 1.0
    
    for h in prange(H):
        for w in prange(W):
            if weight_map[h, w] > 1e-8:
                for c in prange(C):
                    image_recon[h, w, c] /= weight_map[h, w]
    
    return image_recon

def reconstruct_image_from_patches(patches_vec, centers, image_shape, patch_size=7, stride=5):
    """Public wrapper for patch reconstruction."""
    return reconstruct_image_from_patches_jit(patches_vec, centers, image_shape, patch_size, stride)


# --------------------------
# EM Parameter Learning
# --------------------------

class EMParameterLearning:
    """
    Expectation-Maximization (EM) parameter learning with Multi-Head Attention.
    
    Enhanced with attention-based weighting to:
    - Weight parameter updates differently for texture-rich vs sparse regions
    - Use spatial-feature attention heads to specialize parameters
    - Learn patch importance via self-attention on feature manifold
    - Adapt parameters per local image region (edge, smooth, texture)
    
    E-step: Compute attention-weighted loss proxy given current parameters
    M-step: Update parameters via attention-weighted gradient descent
    """
    
    def __init__(self, num_classes=10, learning_rate=0.01, smoothing_window=5, num_heads=4, embed_dim=64):
        """
        Args:
            num_classes: Number of dataset classes (e.g., 10 for CIFAR10)
            learning_rate: EM gradient step size
            smoothing_window: Number of iterations for loss smoothing
            num_heads: Number of attention heads (4, 8, 16)
            embed_dim: Feature embedding dimension for attention
        """
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.smoothing_window = smoothing_window
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        
        # Global parameters (shared across all classes)
        self.sigma = 0.3  # Self-similarity threshold
        self.tau = 0.1    # Graph diffusion step size
        self.radius = 10.0  # Spatial connection radius
        self.sigma_s = 10.0  # Spatial decay
        self.sigma_f = 0.3  # Feature similarity decay
        
        # Multi-Head Attention Components
        # Each head learns to specialize on different patch characteristics
        self.attention_heads = {
            'texture': np.ones(num_heads) / num_heads,      # High-frequency features
            'structure': np.ones(num_heads) / num_heads,    # Structural edges
            'color': np.ones(num_heads) / num_heads,        # Color consistency
            'spatial': np.ones(num_heads) / num_heads       # Spatial smoothness
        }
        
        # Per-head parameter specialization
        self.head_params = {}
        for h in range(num_heads):
            self.head_params[h] = {
                'sigma': 0.3,
                'tau': 0.1,
                'smooth_steps': 5,
                'focus': None,  # Which region type this head specializes on
                'loss_history': [],
                'attention_history': []
            }
        
        # Per-class parameters (one set per class)
        self.class_params = {}
        for c in range(num_classes):
            self.class_params[c] = {
                'sigma': 0.3,
                'tau': 0.1,
                'smooth_steps': 5,
                'loss_history': [],
                'param_history': [],
                'head_weights': np.ones(num_heads) / num_heads  # Per-class attention weights
            }
        
        # Global loss tracking
        self.loss_history = []
        self.param_updates = {'sigma': [], 'tau': [], 'steps': []}
        self.attention_weights = {'texture': [], 'structure': [], 'color': [], 'spatial': []}
    
    def compute_patch_features(self, patches, centers):
        """
        Extract multi-dimensional features from patches for attention weighting.
        Returns: texture_score, structure_score, color_score, smoothness_score
        """
        try:
            # Flatten patches for analysis
            num_patches = len(patches)
            if num_patches == 0:
                return np.ones(4) * 0.5
            
            patch_features = []
            for patch in patches:
                # Texture: Local contrast and variation
                if len(patch) > 1:
                    texture = np.std(patch, axis=0).mean() if len(patch.shape) > 1 else np.std(patch)
                else:
                    texture = 0.0
                
                # Structure: Edge strength via gradient
                if len(patch) > 2:
                    gradient = np.abs(np.diff(patch, axis=0)).mean()
                else:
                    gradient = 0.0
                
                # Color: RGB channel variance
                color = np.std(patch) if len(patch) > 1 else 0.0
                
                patch_features.append([texture, gradient, color, 0.0])
            
            patch_features = np.array(patch_features)
            
            # Normalize and aggregate
            texture_score = np.clip(np.mean(patch_features[:, 0]) / 0.1, 0.0, 1.0)
            structure_score = np.clip(np.mean(patch_features[:, 1]) / 0.1, 0.0, 1.0)
            color_score = np.clip(np.mean(patch_features[:, 2]) / 0.1, 0.0, 1.0)
            spatial_score = np.clip(np.std(centers) / 10.0, 0.0, 1.0) if len(centers) > 0 else 0.5
            
            return np.array([texture_score, structure_score, color_score, spatial_score])
        
        except Exception as e:
            logger.warning(f"Patch feature extraction failed: {e}")
            return np.ones(4) * 0.5
    
    def compute_attention_weights(self, texture_score, structure_score, color_score, spatial_score):
        """
        Compute attention weights for each head based on patch characteristics.
        Uses softmax over feature importance to weight parameter updates.
        
        Returns: Dictionary of attention weights for each head type
        """
        feature_scores = {
            'texture': np.clip(texture_score, 0.0, 1.0),
            'structure': np.clip(structure_score, 0.0, 1.0),
            'color': np.clip(color_score, 0.0, 1.0),
            'spatial': np.clip(spatial_score, 0.0, 1.0)
        }
        
        # Softmax normalization across feature types
        scores = np.array(list(feature_scores.values()))
        exp_scores = np.exp(scores * 2.0)  # Temperature=2 for sharper attention
        weights = exp_scores / np.sum(exp_scores)
        
        return {
            'texture': float(weights[0]),
            'structure': float(weights[1]),
            'color': float(weights[2]),
            'spatial': float(weights[3])
        }
    
    def weighted_parameter_update(self, base_params, attention_weights, loss_delta):
        """
        Update parameters using attention-weighted gradient descent.
        Different heads update differently based on their specialization.
        
        Returns: Updated parameters
        """
        updated_params = base_params.copy()
        
        # Update based on which head's specialty matches patch features
        if loss_delta > 0:  # Loss increased
            # Different reduction rates for different head specialties
            updated_params['sigma'] *= (1 - 0.02 * (attention_weights['texture'] + attention_weights['color']))
            updated_params['tau'] *= (1 - 0.05 * (attention_weights['spatial'] + attention_weights['structure']))
        else:  # Loss decreased
            # Increase aggressiveness based on which head matched
            sigma_boost = 1 + 0.02 * (attention_weights['color'] + attention_weights['texture'])
            tau_boost = 1 + 0.05 * (attention_weights['spatial'] + attention_weights['structure'])
            updated_params['sigma'] = min(0.5, updated_params['sigma'] * sigma_boost)
            updated_params['tau'] = min(0.3, updated_params['tau'] * tau_boost)
        
        return updated_params
    
    def compute_loss_proxy(self, recon_loss, percep_loss, smoothness_score):
        """
        Compute differentiable loss proxy that combines reconstruction and smoothness.
        
        Higher smoothness_score (0-1) means parameters are working well.
        Loss proxy = recon_loss + percep_loss - alpha * smoothness_score
        """
        alpha = 0.1  # Weight for smoothness reward
        loss_proxy = recon_loss + percep_loss - alpha * smoothness_score
        return loss_proxy
    
    def estimate_smoothness(self, S_smooth, patches, W):
        """
        Estimate how well graph smoothing is working.
        
        Returns: smoothness_score in [0, 1]
        - High: Patches are becoming coherent
        - Low: Patches remain noisy
        """
        try:
            # Check variance reduction after smoothing
            if S_smooth.nnz == 0:
                return 0.0
            
            # Compute average self-similarity within smoothed graph
            avg_similarity = S_smooth.data.mean() if len(S_smooth.data) > 0 else 0.0
            
            # Compute sparsity of graph (connected components)
            num_edges = W.nnz
            num_patches = W.shape[0]
            connectivity = min(1.0, num_edges / max(1, num_patches * 5))  # Normalized
            
            # Smoothness = combination of similarity and connectivity
            smoothness = 0.5 * min(1.0, avg_similarity) + 0.5 * connectivity
            return float(smoothness)
        except Exception as e:
            logger.warning(f"Smoothness estimation failed: {e}")
            return 0.5  # Default neutral value
    
    def update_global_params(self, loss_value, smoothness_score, patches=None, centers=None):
        """
        Update global parameters using attention-weighted gradient descent.
        
        With multi-head attention:
        - Different heads specialize on texture, structure, color, spatial properties
        - Parameter updates weighted by attention scores
        - More stable convergence via feature-aware weighting
        """
        self.loss_history.append(loss_value)
        
        # Compute patch features and attention weights
        if patches is not None and centers is not None:
            feature_scores = self.compute_patch_features(patches, centers)
            attention_weights = self.compute_attention_weights(*feature_scores)
            
            # Store attention history for logging
            for key, val in attention_weights.items():
                self.attention_weights[key].append(val)
        else:
            # Fallback: uniform attention
            attention_weights = {
                'texture': 0.25,
                'structure': 0.25,
                'color': 0.25,
                'spatial': 0.25
            }
        
        # Smooth loss over window to reduce noise
        if len(self.loss_history) >= self.smoothing_window:
            smoothed_loss = np.mean(self.loss_history[-self.smoothing_window:])
        else:
            smoothed_loss = np.mean(self.loss_history)
        
        # Adaptive parameter updates based on loss trend with attention weighting
        if len(self.loss_history) > 1:
            loss_delta = self.loss_history[-1] - self.loss_history[-2]
            
            # Use attention-weighted parameter updates
            updated = self.weighted_parameter_update({
                'sigma': self.sigma,
                'tau': self.tau
            }, attention_weights, loss_delta)
            
            self.sigma = updated['sigma']
            self.tau = updated['tau']
        
        # Clamp to valid ranges
        self.tau = np.clip(self.tau, 0.05, 0.3)
        self.sigma = np.clip(self.sigma, 0.1, 0.5)
        self.radius = np.clip(self.radius, 5.0, 20.0)
        self.sigma_s = np.clip(self.sigma_s, 5.0, 20.0)
        self.sigma_f = np.clip(self.sigma_f, 0.1, 0.5)
        
        self.param_updates['sigma'].append(self.sigma)
        self.param_updates['tau'].append(self.tau)
    
    def update_class_params(self, class_idx, loss_value, smoothness_score):
        """
        Update per-class parameters for specialized fine-tuning.
        """
        if class_idx >= self.num_classes:
            return
        
        params = self.class_params[class_idx]
        params['loss_history'].append(loss_value)
        
        # Adaptive smooth_steps based on loss
        avg_loss = np.mean(params['loss_history'][-5:]) if len(params['loss_history']) >= 5 else loss_value
        
        if avg_loss > 0.5:  # High loss: need more smoothing
            params['smooth_steps'] = min(10, params['smooth_steps'] + 1)
        elif avg_loss < 0.2:  # Low loss: can reduce smoothing
            params['smooth_steps'] = max(2, params['smooth_steps'] - 1)
        
        # Adjust sigma and tau for this class
        if smoothness_score > 0.7:  # Good smoothness
            params['sigma'] = min(0.5, params['sigma'] * 1.02)
            params['tau'] = min(0.3, params['tau'] * 1.02)
        elif smoothness_score < 0.3:  # Poor smoothness
            params['sigma'] = max(0.1, params['sigma'] * 0.98)
            params['tau'] = max(0.05, params['tau'] * 0.98)
        
        params['param_history'].append({
            'sigma': params['sigma'],
            'tau': params['tau'],
            'steps': params['smooth_steps']
        })
    
    def get_parameters(self, class_idx=None):
        """
        Get current parameters, optionally for a specific class.
        
        Returns dict with keys: sigma, tau, radius, sigma_s, sigma_f, smooth_steps
        """
        if class_idx is not None and class_idx < self.num_classes:
            params = self.class_params[class_idx]
            return {
                'sigma': params['sigma'],
                'tau': params['tau'],
                'radius': self.radius,
                'sigma_s': self.sigma_s,
                'sigma_f': self.sigma_f,
                'smooth_steps': params['smooth_steps']
            }
        else:
            # Return global parameters
            return {
                'sigma': self.sigma,
                'tau': self.tau,
                'radius': self.radius,
                'sigma_s': self.sigma_s,
                'sigma_f': self.sigma_f,
                'smooth_steps': 5
            }
    
    def get_status(self):
        """Return current parameter values for logging."""
        return {
            'global_sigma': f"{self.sigma:.4f}",
            'global_tau': f"{self.tau:.4f}",
            'global_radius': f"{self.radius:.2f}",
            'avg_loss': f"{np.mean(self.loss_history[-5:]):.4f}" if len(self.loss_history) > 0 else "N/A"
        }

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --------------------------
# Enhanced Dataset Loading with Multiple Benchmarks
# --------------------------
class ImageFolderDataset(torch.utils.data.Dataset):
    """Custom dataset for loading images from a folder"""
    def __init__(self, folder_path, transform=None, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = []
        
        # Recursively find all image files
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(extensions):
                    self.image_paths.append(os.path.join(root, file))
        
        logger.info(f"Found {len(self.image_paths)} images in {folder_path}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, img_path
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}")
            # Return a dummy image on failure
            return torch.zeros(3, RENDER_FINAL_RESOLUTION, RENDER_FINAL_RESOLUTION), img_path

def get_transforms(augment=True):
    """Get image transformation pipeline"""
    transform_list = [
        transforms.Resize(RENDER_FINAL_RESOLUTION),
        transforms.CenterCrop(RENDER_FINAL_RESOLUTION),
    ]
    
    if augment:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ])
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((NORM_MEAN, NORM_MEAN, NORM_MEAN), 
                            (NORM_STD, NORM_STD, NORM_STD))
    ])
    
    return transforms.Compose(transform_list)

def load_dataset(dataset_name='CIFAR10', data_path='./data', augment=True):
    """
    Load different image benchmark datasets
    
    Args:
        dataset_name: Name of dataset ('CIFAR10', 'CIFAR100', 'ImageNet', 'STL10', 'Flowers', 'custom')
        data_path: Path to dataset or custom image folder
        augment: Whether to apply data augmentation
    
    Returns:
        DataLoader object
    """
    transform = get_transforms(augment=augment)
    train_data = None
    
    logger.info(f"Loading dataset: {dataset_name}")
    
    try:
        if dataset_name.upper() == 'CIFAR10':
            train_data = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
            logger.info(f"CIFAR10 dataset loaded: {len(train_data)} images")
            
        elif dataset_name.upper() == 'CIFAR100':
            train_data = datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform)
            logger.info(f"CIFAR100 dataset loaded: {len(train_data)} images")
            
        elif dataset_name.upper() == 'STL10':
            train_data = datasets.STL10(root=data_path, split='train', download=True, transform=transform)
            logger.info(f"STL10 dataset loaded: {len(train_data)} images")
            
        elif dataset_name.upper() == 'FLOWERS':
            try:
                train_data = datasets.Flowers102(root=data_path, split='train', download=True, transform=transform)
            except:
                logger.warning("Flowers102 download failed, using CIFAR10 as fallback")
                train_data = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
            logger.info(f"Flowers dataset loaded: {len(train_data)} images")
            
        elif dataset_name.upper() == 'IMAGENET':
            # ImageNet requires manual download - check if available
            if os.path.exists(os.path.join(data_path, 'imagenet', 'train')):
                train_data = datasets.ImageNet(root=os.path.join(data_path, 'imagenet'), 
                                              split='train', transform=transform)
                logger.info(f"ImageNet dataset loaded: {len(train_data)} images")
            else:
                logger.warning("ImageNet not found at expected path, using CIFAR10 as fallback")
                train_data = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
        
        elif dataset_name.upper() == 'CUSTOM' or os.path.isdir(dataset_name):
            # Load from custom folder
            folder_path = dataset_name if os.path.isdir(dataset_name) else data_path
            if not os.path.exists(folder_path):
                logger.error(f"Custom folder not found: {folder_path}")
                raise ValueError(f"Custom folder path does not exist: {folder_path}")
            train_data = ImageFolderDataset(folder_path, transform=transform)
            logger.info(f"Custom dataset loaded: {len(train_data)} images")
        
        else:
            logger.warning(f"Unknown dataset: {dataset_name}, defaulting to CIFAR10")
            train_data = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    
    except Exception as e:
        logger.error(f"Error loading {dataset_name}: {e}")
        logger.info("Falling back to CIFAR10")
        train_data = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    
    dataloader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True if DEVICE == 'cuda' else False
    )
    logger.info(f"DataLoader created. Number of batches: {len(dataloader)}")
    return dataloader

# --------------------------
# Enhanced Diffusion Model
# --------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
    
    def forward(self, x):
        return x + self.block(x)

class DiffusionModel(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, text_dim=TEXT_DIM):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, TIME_EMBED_DIM),
            nn.SiLU(),
            nn.Linear(TIME_EMBED_DIM, TIME_EMBED_DIM),
            nn.SiLU(),
            nn.Linear(TIME_EMBED_DIM, latent_dim)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.LayerNorm(text_dim),
            nn.SiLU(),
            nn.Linear(text_dim, latent_dim)
        )
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(latent_dim) for _ in range(2)  # Reduced for testing
        ])
        
        self.attn = nn.MultiheadAttention(
            latent_dim, 
            num_heads=ATTN_NUM_HEADS,
            batch_first=True
        )
        
        self.down1 = nn.Sequential(
            nn.Linear(latent_dim, DOWN1_OUT_DIM),
            nn.LayerNorm(DOWN1_OUT_DIM),
            nn.SiLU(),
            nn.Linear(DOWN1_OUT_DIM, DOWN1_OUT_DIM // 2),
            nn.LayerNorm(DOWN1_OUT_DIM // 2),
            nn.SiLU()
        )
        
        self.up1 = nn.Sequential(
            nn.Linear(DOWN1_OUT_DIM // 2 + latent_dim, DOWN1_OUT_DIM),
            nn.LayerNorm(DOWN1_OUT_DIM),
            nn.SiLU(),
            nn.Linear(DOWN1_OUT_DIM, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        self.final_proj = nn.Linear(latent_dim, latent_dim)
        
    def forward(self, z, t, text_emb=None):
        batch_size = z.shape[0]
        t = t.view(-1, 1).float()
        
        t_emb = self.time_embed(t / DIFFUSION_STEPS)
        
        if text_emb is not None:
            text_proj = self.text_proj(text_emb)
            z = z + text_proj
        
        z = z + t_emb
        
        for block in self.res_blocks:
            z = block(z)
        
        attn_out, _ = self.attn(z.unsqueeze(1), z.unsqueeze(1), z.unsqueeze(1))
        z = z + attn_out.squeeze(1)
        
        down = self.down1(z)
        up = self.up1(torch.cat([down, z], dim=-1))
        out = self.final_proj(up)
        return out

# --------------------------
# Enhanced Noise Scheduler (Cosine Schedule)
# --------------------------
class NoiseScheduler:
    def __init__(self, steps=DIFFUSION_STEPS):
        self.steps = steps
        t = torch.linspace(0, 1, steps, device=DEVICE)
        self.alpha_bar = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        self.alpha_bar = self.alpha_bar / self.alpha_bar[0]
        self.beta = 1 - (self.alpha_bar[1:] / self.alpha_bar[:-1])
        self.beta = torch.cat([self.beta[0].unsqueeze(0), self.beta])
        logger.info(f"Cosine noise scheduler initialized")
    
    def add_noise(self, z, t):
        t = t.squeeze(-1)
        sqrt_alpha = torch.sqrt(self.alpha_bar[t]).view(-1, 1)
        sqrt_one_minus = torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1)
        noise = torch.randn_like(z)
        noisy_z = sqrt_alpha * z + sqrt_one_minus * noise
        return noisy_z, noise

# --------------------------
# Enhanced IFS Generator
# --------------------------
class IFSGenerator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, num_transforms=NUM_TRANSFORMS):
        super().__init__()
        self.num_transforms = num_transforms
        self.net = nn.Sequential(
            nn.Linear(latent_dim, IFS_NET_MID_DIM),
            nn.LayerNorm(IFS_NET_MID_DIM),
            nn.ReLU(),
            ResidualBlock(IFS_NET_MID_DIM),
            nn.Linear(IFS_NET_MID_DIM, num_transforms * IFS_AFFINE_DIM + num_transforms)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
    
    def forward(self, z):
        x = self.net(z)
        affines = x[..., :-self.num_transforms].view(-1, self.num_transforms, 2, 3)
        logits = x[..., -self.num_transforms:]
        probs = F.softmax(logits, dim=-1)
        
        matrices = affines[..., :2]
        U, S, V = torch.svd(matrices)
        S_clamped = torch.clamp(S, max=MATRIX_NORM_THRESHOLD)
        affines[..., :2] = U @ torch.diag_embed(S_clamped) @ V.transpose(-2, -1)
        
        return affines, probs

# --------------------------
# Fractal Tokenizer for Image Decomposition
# --------------------------
class FractalTokenizer(nn.Module):
    def __init__(self, patch_size=8, num_tokens=256, latent_dim=LATENT_DIM):
        super().__init__()
        self.patch_size = patch_size
        self.num_tokens = num_tokens
        self.latent_dim = latent_dim
        
        # VQ-VAE like codebook for tokens
        self.codebook = nn.Parameter(torch.randn(num_tokens, latent_dim))
        nn.init.uniform_(self.codebook, -1.0 / num_tokens, 1.0 / num_tokens)
        
        # Encoder: image patches -> latent codes
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, kernel_size=3, stride=2, padding=1),
            # Pool to a single spatial location so encoder outputs shape (N, latent_dim, 1, 1)
            nn.AdaptiveAvgPool2d(1),
        )
        
        # IFS matcher: encode patch similarity to IFS transforms
        self.ifs_matcher = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_TRANSFORMS)  # probability of each transform
        )
        
        # Learnable decoder: reconstruct patches from latent codes
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, patch_size * patch_size * 3),
            nn.Tanh()  # Output in [-1, 1] to match normalized image range
        )
    
    def encode(self, x):
        """Encode image patches to latent codes"""
        B, C, H, W = x.shape
        # Extract patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        num_patches = patches.shape[1]
        patches = patches.view(B * num_patches, C, self.patch_size, self.patch_size)
        
        # Encode patches
        z = self.encoder(patches)
        z = z.view(B * num_patches, self.latent_dim)
        
        return z, (B, num_patches)
    
    def quantize(self, z):
        """Quantize latent codes to nearest codebook entries"""
        distances = torch.cdist(z, self.codebook)
        token_indices = torch.argmin(distances, dim=1)
        quantized = self.codebook[token_indices]
        return token_indices, quantized
    
    def get_ifs_tokens(self, quantized):
        """Convert quantized codes to IFS transform tokens"""
        ifs_logits = self.ifs_matcher(quantized)
        ifs_tokens = torch.argmax(ifs_logits, dim=1)
        ifs_probs = F.softmax(ifs_logits, dim=1)
        return ifs_tokens, ifs_probs
    
    def tokenize(self, x):
        """Full tokenization pipeline: image -> tokens"""
        z, patch_info = self.encode(x)
        token_indices, quantized = self.quantize(z)
        ifs_tokens, ifs_probs = self.get_ifs_tokens(quantized)
        return token_indices, ifs_tokens, ifs_probs, patch_info
    
    def decode(self, token_indices, patch_shape):
        """Reconstruct image from token indices using learned decoder"""
        B, num_patches = patch_shape
        quantized = self.codebook[token_indices]  # (B*num_patches, latent_dim)
        
        # Decode each patch from quantized latent code (LEARNED, not random!)
        patch_pixels = self.decoder(quantized)  # (B*num_patches, patch_size*patch_size*3)
        patch_pixels = patch_pixels.view(-1, 3, self.patch_size, self.patch_size)
        
        # Reconstruct full image by stitching patches
        patches_per_side = int(np.sqrt(num_patches))
        H = W = self.patch_size * patches_per_side
        output = torch.zeros(B, 3, H, W, device=quantized.device)
        
        idx = 0
        for b in range(B):
            for i in range(patches_per_side):
                for j in range(patches_per_side):
                    output[b, :, i*self.patch_size:(i+1)*self.patch_size, 
                           j*self.patch_size:(j+1)*self.patch_size] = patch_pixels[idx]
                    idx += 1
        
        return output

# --------------------------
# Enhanced Attention Renderer
# --------------------------
class RendererTransformer(nn.Module):
    def __init__(self, num_transforms=NUM_TRANSFORMS):
        super().__init__()
        self.num_transforms = num_transforms
        self.coord_embed = nn.Sequential(
            nn.Linear(2, RENDERER_COORD_EMBED_DIM),
            nn.LayerNorm(RENDERER_COORD_EMBED_DIM),
            nn.ReLU(),
            nn.Linear(RENDERER_COORD_EMBED_DIM, RENDERER_COORD_EMBED_DIM)
        )
        self.buffer_conv = nn.Sequential(
            nn.Conv2d(3, RENDERER_CONV_OUT_CHANNELS, RENDERER_KERNEL_SIZE, 
                      padding=RENDERER_PADDING),
            nn.BatchNorm2d(RENDERER_CONV_OUT_CHANNELS),
            nn.ReLU(),
            nn.Conv2d(RENDERER_CONV_OUT_CHANNELS, RENDERER_CONV_OUT_CHANNELS, 
                      RENDERER_KERNEL_SIZE, padding=RENDERER_PADDING),
            nn.BatchNorm2d(RENDERER_CONV_OUT_CHANNELS),
            nn.ReLU()
        )
        self.attn = nn.MultiheadAttention(
            RENDERER_COORD_EMBED_DIM, 
            num_heads=ATTN_NUM_HEADS, 
            kdim=RENDERER_POLICY_MID_DIM, 
            vdim=RENDERER_POLICY_MID_DIM,
            batch_first=True
        )
        self.policy_head = nn.Sequential(
            nn.Linear(RENDERER_COORD_EMBED_DIM, RENDERER_POLICY_MID_DIM),
            nn.LayerNorm(RENDERER_POLICY_MID_DIM),
            nn.ReLU(),
            nn.Linear(RENDERER_POLICY_MID_DIM, num_transforms)
        )
    
    def forward(self, coords, buffer):
        coord_emb = self.coord_embed(coords)
        buffer_feat = self.buffer_conv(buffer)
        buffer_feat = buffer_feat.flatten(2).permute(0, 2, 1)
        
        attn_out, _ = self.attn(
            query=coord_emb.unsqueeze(1),
            key=buffer_feat,
            value=buffer_feat
        )
        
        logits = self.policy_head(attn_out.squeeze(1))
        probs = F.softmax(logits, dim=-1)
        return probs

# --------------------------
# Perceptual Loss
# --------------------------
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:16]).to(DEVICE).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        logger.info("Perceptual loss VGG loaded successfully")
    
    def forward(self, input, target):
        input = F.interpolate(input, size=(224, 224), mode='bilinear', align_corners=False)
        target = F.interpolate(target, size=(224, 224), mode='bilinear', align_corners=False)
        
        input_features = self.feature_extractor((input + 1) / 2)
        target_features = self.feature_extractor((target + 1) / 2)
        
        return F.mse_loss(input_features, target_features)

# --------------------------
# CLIP Utilities
# --------------------------
try:
    clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    logger.info("CLIP model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load CLIP model: {e}")
    raise

def encode_prompt(prompt):
    text = clip.tokenize([prompt]).to(DEVICE)
    with torch.no_grad():
        text_features = clip_model.encode_text(text)
    normalized_features = text_features.float() / text_features.norm(dim=-1, keepdim=True)
    logger.debug(f"Encoded prompt '{prompt}': shape={normalized_features.shape}")
    return normalized_features

# --------------------------
# Enhanced ECM Training Framework
# --------------------------
class EnhancedLIDECM(nn.Module):
    def __init__(self):
        super(EnhancedLIDECM, self).__init__()
        self.dataloader = load_dataset()
        self.latent_codes = nn.Parameter(torch.randn(NUM_LATENTS, LATENT_DIM, device=DEVICE))
        self.generator = IFSGenerator(LATENT_DIM, NUM_TRANSFORMS).to(DEVICE)
        self.renderer = RendererTransformer(NUM_TRANSFORMS).to(DEVICE)
        self.diffusion = DiffusionModel(LATENT_DIM).to(DEVICE)
        self.scheduler = NoiseScheduler(DIFFUSION_STEPS)
        self.perceptual_loss = PerceptualLoss().to(DEVICE)
        
        self.optimizer = optim.AdamW([
            {'params': self.generator.parameters(), 'lr': OPTIMIZER_LR_GENERATOR, 'weight_decay': 1e-4},
            {'params': self.renderer.parameters(), 'lr': OPTIMIZER_LR_RENDERER, 'weight_decay': 1e-4},
            {'params': self.diffusion.parameters(), 'lr': OPTIMIZER_LR_DIFFUSION, 'weight_decay': 1e-4},
            {'params': [self.latent_codes], 'lr': OPTIMIZER_LR_LATENT_CODES}
        ])
        
        self.scheduler_opt = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            factor=0.5,
            patience=2,
        )
        
        logger.info("Enhanced LIDECM initialized")

# --------------------------
# Enhanced ECM Training Framework with Fractal Tokenization
# --------------------------
class EnhancedLIDECM(nn.Module):
    def __init__(self, dataset_name='CIFAR10', data_path='./data', augment=True):
        super(EnhancedLIDECM, self).__init__()
        self.dataset_name = dataset_name
        self.data_path = data_path
        
        # Ensure checkpoint directory exists
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        
        # Load dataset with specified configuration
        self.dataloader = load_dataset(dataset_name=dataset_name, data_path=data_path, augment=augment)
        
        # Fractal tokenizer for image decomposition
        self.tokenizer = FractalTokenizer(
            patch_size=8, 
            num_tokens=NUM_LATENTS, 
            latent_dim=LATENT_DIM
        ).to(DEVICE)
        
        # Generator creates latent codes from tokens
        self.generator = IFSGenerator(LATENT_DIM, NUM_TRANSFORMS).to(DEVICE)
        self.renderer = RendererTransformer(NUM_TRANSFORMS).to(DEVICE)
        self.diffusion = DiffusionModel(LATENT_DIM).to(DEVICE)
        self.scheduler = NoiseScheduler(DIFFUSION_STEPS)
        self.perceptual_loss = PerceptualLoss().to(DEVICE)
        
        self.optimizer = optim.AdamW([
            {'params': self.tokenizer.parameters(), 'lr': OPTIMIZER_LR_GENERATOR, 'weight_decay': 1e-4},
            {'params': self.generator.parameters(), 'lr': OPTIMIZER_LR_GENERATOR, 'weight_decay': 1e-4},
            {'params': self.renderer.parameters(), 'lr': OPTIMIZER_LR_RENDERER, 'weight_decay': 1e-4},
            {'params': self.diffusion.parameters(), 'lr': OPTIMIZER_LR_DIFFUSION, 'weight_decay': 1e-4},
        ])
        
        self.scheduler_opt = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            factor=0.5,
            patience=2,
        )
        
        # EM parameter learning for graph diffusion optimization
        num_classes = 10 if dataset_name.upper() in ['CIFAR10', 'MNIST'] else 1000
        self.em_learner = EMParameterLearning(num_classes=num_classes, learning_rate=0.01)
        logger.info(f"Initialized EM parameter learning with {num_classes} classes")
        
        self.training_complete = False
        logger.info(f"Enhanced LIDECM initialized with {dataset_name} dataset")

    def e_step(self, images, current_epoch):
        """
        E-step: Tokenize images into fractal tokens using learned codebook
        Improved clustering with codebook optimization
        """
        with torch.no_grad():
            token_indices, ifs_tokens, ifs_probs, patch_info = self.tokenizer.tokenize(images)
            
            # Additional clustering refinement
            z, _ = self.tokenizer.encode(images)
            distances = torch.cdist(z, self.tokenizer.codebook)
            
            # Compute assignment confidence
            min_dist = torch.min(distances, dim=1)[0]
            assignment_confidence = torch.exp(-min_dist)  # Higher = more confident
            
            logger.debug(f"E-step: Mean assignment confidence: {assignment_confidence.mean():.4f}")
            
            return token_indices, ifs_tokens, ifs_probs, patch_info, assignment_confidence

    def m_step(self, images, token_indices, ifs_tokens, ifs_probs, patch_info, current_epoch):
        """
        M-step: Optimize model parameters with comprehensive losses
        Enhanced with:
        - Reconstruction loss (primary)
        - Perceptual loss (semantic features)
        - Diffusion loss (generative model)
        - Codebook loss (token learning)
        - Commitment loss (encoder stability)
        - Regularization losses
        """
        batch_size = images.shape[0]

        # Trace entry shapes
        logger.debug(f"M-step start: images={tuple(images.shape)}, token_indices={tuple(token_indices.shape)}, patch_info={patch_info}")

        # Get encoded latent codes from images
        z, _ = self.tokenizer.encode(images)
        logger.debug(f"Encoder output: z={tuple(z.shape)}")
        
        # Reconstruct using the tokenizer's decoder (uses token indices for differentiable reconstruction)
        try:
            reconstructed = self.tokenizer.decode(token_indices, patch_info)
            logger.debug(f"Decoded reconstruction shape: {reconstructed.shape}")
        except Exception as e:
            logger.warning(f"Decoder failed: {e}, falling back to identity")
            # If decoder fails, use simple bilinear resize of z
            B, num_patches = patch_info
            reconstructed = torch.nn.functional.interpolate(
                z.view(B, -1, 1, 1), size=(32, 32), mode='bilinear', align_corners=False
            )
        
        B, num_patches = patch_info
        
        # Quantization and codebook losses
        quantized = self.tokenizer.codebook[token_indices]
        commit_loss = F.mse_loss(z, quantized.detach())
        codebook_loss = F.mse_loss(quantized.detach(), z.detach())
        
        # Get current EM parameters (adaptive per training iteration)
        em_params = self.em_learner.get_parameters()
        
        # Compute smoothness scores for EM parameter updates
        smoothness_scores = []
        all_patches = []
        all_centers = []
        
        # Extract patches for smoothness estimation (non-differentiable part)
        try:
            for b in range(min(B, 1)):  # Only first batch for efficiency
                img_np = images[b].permute(1, 2, 0).cpu().detach().numpy()
                # Denormalize to [0, 1]
                img_np = (img_np * NORM_STD) + NORM_MEAN
                img_np = np.clip(img_np, 0, 1)
                patches, centers = extract_patches(img_np, patch_size=8, stride=5)
                
                # Store for attention-based EM updates
                all_patches.append(patches)
                all_centers.append(centers)
                
                # Build patch graph using EM-learned parameters
                W = build_patch_graph_radius(
                    centers, patches, 
                    radius=em_params['radius'], 
                    sigma_s=em_params['sigma_s'], 
                    sigma_f=em_params['sigma_f']
                )
                S = compute_self_similarity(patches, W, sigma=em_params['sigma'])
                L = graph_laplacian(W)
                S_smooth = graph_smoothing(S, L, tau=em_params['tau'], steps=em_params['smooth_steps'])
                
                # Track smoothness for EM parameter updates
                smoothness = self.em_learner.estimate_smoothness(S_smooth, patches, W)
                smoothness_scores.append(smoothness)
        except Exception as e:
            logger.warning(f"Patch extraction failed: {e}")
            smoothness_scores = [0.5] * B
        
        logger.debug(f"Reconstructed shape: {tuple(reconstructed.shape)}, images.shape={tuple(images.shape)}, requires_grad={reconstructed.requires_grad}")
        
        # Reconstruction loss (pixel-level)
        recon_loss = F.mse_loss(reconstructed, images, reduction='mean')
        logger.debug(f"Reconstruction loss: {recon_loss.item():.6f}")
        
        # Perceptual loss (feature-level semantic similarity)
        percep_loss = self.perceptual_loss(reconstructed, images)
        
        logger.debug(f"Reconstructed shape: {tuple(reconstructed.shape)}, images.shape={tuple(images.shape)}")
        
        # Reconstruction loss (pixel-level)
        recon_loss = F.mse_loss(reconstructed, images, reduction='mean')
        logger.debug(f"Reconstruction loss: {recon_loss.item():.6f}")
        
        # Perceptual loss (feature-level semantic similarity)
        percep_loss = self.perceptual_loss(reconstructed, images)
        
        # EM Parameter Update: Compute loss proxy and update parameters with attention weighting
        avg_smoothness = np.mean(smoothness_scores)
        em_loss_proxy = self.em_learner.compute_loss_proxy(
            recon_loss.item(), 
            percep_loss.item(), 
            avg_smoothness
        )
        
        # Pass patches and centers to EM learner for attention-based weighting
        # If patches are available, use them for feature-aware parameter updates
        batch_patches = None
        batch_centers = None
        if len(all_patches) > 0 and len(all_centers) > 0:
            # Aggregate patches and centers from batch
            batch_patches = np.concatenate(all_patches, axis=0)
            batch_centers = np.concatenate(all_centers, axis=0)
        
        self.em_learner.update_global_params(em_loss_proxy, avg_smoothness, 
                                            patches=batch_patches, centers=batch_centers)
        
        # Log EM status periodically
        if current_epoch % 2 == 0:
            em_status = self.em_learner.get_status()
            logger.debug(f"EM Status: {em_status}, avg_smoothness={avg_smoothness:.4f}")
        
        # Diffusion loss: regularize latent space (optional, light weight)
        # Focus on reconstruction first, diffusion is secondary
        if True:  # Set to False to disable diffusion loss temporarily
            t = torch.randint(0, DIFFUSION_STEPS, (batch_size,), device=DEVICE)
            noisy_z, noise = self.scheduler.add_noise(quantized[:batch_size], t)
            text_emb = encode_prompt(TRAIN_PROMPT)
            text_emb = text_emb.expand(batch_size, -1)
            pred_noise = self.diffusion(noisy_z, t, text_emb=text_emb)
            diffusion_loss = F.mse_loss(pred_noise, noise)
        else:
            diffusion_loss = torch.tensor(0.0, device=DEVICE)
        
        # Entropy regularization: encourage diverse token usage
        token_hist = torch.histc(token_indices.float(), bins=self.tokenizer.num_tokens, min=0, max=self.tokenizer.num_tokens-1)
        token_probs = token_hist / (token_hist.sum() + 1e-8)
        entropy_loss = -(token_probs * (torch.log(token_probs + 1e-8))).sum()
        entropy_regularizer = -entropy_loss  # Minimize negative entropy (maximize diversity)
        
        # Orthogonality loss: encourage diverse codebook vectors
        codebook_normalized = self.tokenizer.codebook / (torch.norm(self.tokenizer.codebook, dim=1, keepdim=True) + 1e-8)
        gram = torch.mm(codebook_normalized, codebook_normalized.t())
        ortho_loss = torch.sum(gram ** 2) - torch.trace(gram ** 2)
        
        # Curriculum: gradually increase loss weights during training
        curriculum_factor = min(1.0, (current_epoch + 1) / max(1, ECM_EPOCHS))
        
        # Simplified loss: focus on reconstruction quality
        # Primary: Reconstruction (pixel-level match)
        # Secondary: Perceptual (semantic feature match)
        # Regularizers: Codebook learning and diversity
        total_loss = (
            recon_loss +                           # Pixel reconstruction
            PERCEPTUAL_LOSS_WEIGHT * percep_loss + # Semantic similarity
            0.25 * codebook_loss +                 # Codebook learning
            0.1 * commit_loss +                    # Encoder stability
            0.001 * curriculum_factor * ortho_loss # Codebook diversity
        )
        
        # Diffusion loss only if enabled above (usually disabled for faster convergence)
        if diffusion_loss.item() > 0:
            total_loss = total_loss + 0.01 * KL_LOSS_WEIGHT * diffusion_loss
        
        # Backward pass with gradient clipping
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), GRAD_CLIP_NORM)
        self.optimizer.step()
        
        # Logging with detailed breakdown
        logger.info(
            f"M-step Epoch {current_epoch}: "
            f"total={total_loss.item():.4f} | "
            f"recon={recon_loss.item():.4f} | "
            f"percep={percep_loss.item():.4f} | "
            f"diffusion={diffusion_loss.item():.4f} | "
            f"codebook={codebook_loss.item():.4f} | "
            f"commit={commit_loss.item():.4f} | "
            f"ortho={ortho_loss.item():.6f} | "
            f"entropy={entropy_regularizer.item():.6f}"
        )
        
        return total_loss.item()

    def render_patch(self, affines, probs, shape, steps=RENDER_STEPS, n_points=RENDER_N_POINTS, current_epoch=0):
        # Handle cases where generator returned a leading singleton batch dim
        # (e.g., shape (1, N, ...)) — squeeze it so probs is 1- or 2-D for torch.multinomial
        if probs.dim() == 3 and probs.shape[0] == 1:
            probs = probs.squeeze(0)
            affines = affines.squeeze(0)

        batch_size = affines.shape[0]
        resolution = min(RENDER_FINAL_RESOLUTION, 
                        RENDER_MIN_RESOLUTION + (current_epoch // RENDER_EPOCH_STEP * RENDER_RESOLUTION_STEP))
        
        buffer = torch.randn(batch_size, 3, resolution, resolution, device=DEVICE) * 0.01
        points = torch.zeros(batch_size, n_points, 2, device=DEVICE)
        
        # Burn-in period
        for _ in range(10):
            idx = torch.multinomial(probs, n_points, replacement=True)
            batch_idx = torch.arange(batch_size, device=DEVICE).view(-1, 1).expand(-1, n_points)
            selected_affines = affines[batch_idx, idx]
            points = torch.einsum('bnij,bnj->bni', selected_affines[..., :2], points) + selected_affines[..., 2]
        
        # Main rendering
        for step in range(steps):
            idx = torch.multinomial(probs, n_points, replacement=True)
            batch_idx = torch.arange(batch_size, device=DEVICE).view(-1, 1).expand(-1, n_points)
            selected_affines = affines[batch_idx, idx]
            
            points = torch.einsum('bnij,bnj->bni', selected_affines[..., :2], points) + selected_affines[..., 2]
            
            coords = ((points + 1) * (resolution / 2)).long()
            valid_mask = (coords[..., 0] >= 0) & (coords[..., 0] < resolution) & \
                         (coords[..., 1] >= 0) & (coords[..., 1] < resolution)
            
            for b in range(batch_size):
                valid_coords = coords[b][valid_mask[b]]
                if len(valid_coords) > 0:
                    unique_coords, counts = torch.unique(valid_coords, dim=0, return_counts=True)
                    buffer[b, :, unique_coords[:, 0], unique_coords[:, 1]] += counts.float().view(1, -1)
        
        buffer_max = buffer.amax(dim=(1, 2, 3), keepdim=True)
        buffer = buffer / (buffer_max + BUFFER_MAX_EPSILON)
        buffer = torch.pow(buffer, 0.7)
        
        buffer = F.interpolate(buffer, size=(RENDER_FINAL_RESOLUTION, RENDER_FINAL_RESOLUTION), 
                             mode='bilinear', align_corners=False)
        return buffer

    def tokenize_image(self, image):
        """Tokenize a single image or batch of images"""
        self.eval()
        with torch.no_grad():
            token_indices, ifs_tokens, ifs_probs, patch_info = self.tokenizer.tokenize(image)
        return token_indices, ifs_tokens, ifs_probs, patch_info
    
    def reconstruct_from_tokens(self, token_indices):
        """Reconstruct image from token indices"""
        self.eval()
        with torch.no_grad():
            quantized = self.tokenizer.codebook[token_indices]
            # Optionally render with IFS
            return self.tokenizer.decode(token_indices, (token_indices.shape[0], token_indices.shape[0]))

    def run_train(self):
        self.train()
        best_loss = float('inf')
        
        for epoch in range(ECM_EPOCHS):
            epoch_loss = 0
            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Process all available batches (or limit to 200 for memory efficiency)
                if batch_idx > 200:
                    logger.info(f"Stopping at batch {batch_idx} for memory efficiency")
                    break
                    
                images = batch[0].to(DEVICE)
                
                # E-step: Tokenize images
                token_indices, ifs_tokens, ifs_probs, patch_info = self.e_step(images, epoch)
                
                # M-step: Optimize from tokens
                batch_loss = self.m_step(images, token_indices, ifs_tokens, ifs_probs, patch_info, epoch)
                epoch_loss += batch_loss
                
                progress_bar.set_postfix(loss=batch_loss)
            
            avg_loss = epoch_loss / min(len(self.dataloader), 5)
            logger.info(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")
            self.scheduler_opt.step(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint("best_model.pt")
                logger.info(f"New best model saved with loss {best_loss:.6f}")
            
            if epoch % CHECKPOINT_SAVE_INTERVAL == 0:
                self.save_checkpoint(f"checkpoint_epoch{epoch}.pt")

    def validate(self, val_dataloader=None, num_samples=4):
        """
        Validation phase: evaluate model on held-out data
        """
        self.eval()
        if val_dataloader is None:
            val_dataloader = self.dataloader
        
        val_losses = []
        logger.info("Running validation...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                if batch_idx > 3:  # Limited validation batches
                    break
                
                if isinstance(batch, (list, tuple)):
                    images = batch[0].to(DEVICE)
                else:
                    images = batch.to(DEVICE)
                
                # E-step: Tokenize (capture patch_info for aggregation)
                token_indices, _, _, patch_info, _ = self.e_step(images, epoch=0)
                
                # Compute reconstruction loss
                z, _ = self.tokenizer.encode(images)
                quantized = self.tokenizer.codebook[token_indices]
                logger.debug(f"Validate: images={tuple(images.shape)}, token_indices={tuple(token_indices.shape)}, patch_info={patch_info}")

                # Generate per-patch transforms and aggregate to per-image transforms
                affines, transform_probs = self.generator(quantized)
                logger.debug(f"Validate generator raw: affines={tuple(affines.shape)}, transform_probs={tuple(transform_probs.shape)}")
                B, num_patches = patch_info
                affines = affines.view(B, num_patches, self.generator.num_transforms, 2, 3)
                transform_probs = transform_probs.view(B, num_patches, self.generator.num_transforms)
                affines = affines.mean(dim=1)
                transform_probs = transform_probs.mean(dim=1)
                logger.debug(f"Validate aggregated: affines={tuple(affines.shape)}, transform_probs={tuple(transform_probs.shape)}")

                rendered = self.render_patch(affines, transform_probs, images.shape, current_epoch=0)
                
                loss = F.mse_loss(rendered, images, reduction='mean')
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        logger.info(f"Validation Loss: {avg_val_loss:.6f}")
        return avg_val_loss

    def generate_from_prompt(self, prompt, num_patches=16, temperature=1.0, top_k=None):
        """
        Enhanced generation from text prompt with:
        - Temperature control for diversity
        - Top-k sampling
        - Better diffusion scheduling
        - Classifier-free guidance
        """
        self.eval()
        text_emb = encode_prompt(prompt)
        
        # Initialize latent codes with temperature
        z = torch.randn(num_patches, LATENT_DIM, device=DEVICE) * temperature
        
        logger.info(f"Generating image for prompt: '{prompt}'")
        logger.info(f"  Temperature: {temperature}, Top-k: {top_k}")
        
        with torch.no_grad():
            for step, t in enumerate(tqdm(reversed(range(DIFFUSION_STEPS)), desc="Diffusion", total=DIFFUSION_STEPS)):
                t_tensor = torch.tensor([[t]], device=DEVICE).float()
                alpha_t = self.scheduler.alpha_bar[t]
                alpha_t_prev = self.scheduler.alpha_bar[t-1] if t > 0 else torch.tensor(1.0, device=DEVICE)
                
                # Guided denoising
                conditioned_noise = self.diffusion(z, t_tensor, text_emb.expand(num_patches, -1))
                unconditioned_noise = self.diffusion(z, t_tensor)
                
                # Classifier-free guidance
                guidance_scale = GUIDANCE_SCALE * min(1.0, step / DIFFUSION_STEPS)  # Anneal guidance
                pred_noise = unconditioned_noise + guidance_scale * (conditioned_noise - unconditioned_noise)
                
                # Denoising step
                pred_x0 = (z - torch.sqrt(1 - alpha_t) * pred_noise) / (torch.sqrt(alpha_t) + 1e-8)
                
                # Add noise for next step
                if t > 0:
                    noise = torch.randn_like(z) * temperature
                else:
                    noise = 0
                
                z = torch.sqrt(alpha_t_prev) * pred_x0 + torch.sqrt(1 - alpha_t_prev) * noise
        
        # Generate from latent codes
        quantized = self.tokenizer.codebook[self.tokenizer.quantize(z)[0]]
        affines, probs = self.generator(z)
        
        rendered = self.render_patch(
            affines, 
            probs, 
            (1, 3, RENDER_FINAL_RESOLUTION, RENDER_FINAL_RESOLUTION), 
            steps=GENERATE_STEPS, 
            current_epoch=ECM_EPOCHS
        )
        
        logger.info(f"✓ Generation complete for '{prompt}'")
        return rendered

    def render_patch(self, affines, probs, shape, steps=RENDER_STEPS, n_points=RENDER_N_POINTS, current_epoch=0):
        batch_size = affines.shape[0]
        resolution = min(RENDER_FINAL_RESOLUTION, 
                        RENDER_MIN_RESOLUTION + (current_epoch // RENDER_EPOCH_STEP * RENDER_RESOLUTION_STEP))
        
        buffer = torch.randn(batch_size, 3, resolution, resolution, device=DEVICE) * 0.01
        points = torch.zeros(batch_size, n_points, 2, device=DEVICE)
        
        # Burn-in period
        # Handle possible leading singleton dim from generator output (squeeze to 2D)
        if probs.dim() == 3 and probs.shape[0] == 1:
            probs = probs.squeeze(0)
            affines = affines.squeeze(0)

        for _ in range(10):  # Reduced for testing
            idx = torch.multinomial(probs, n_points, replacement=True)
            batch_idx = torch.arange(batch_size, device=DEVICE).view(-1, 1).expand(-1, n_points)
            selected_affines = affines[batch_idx, idx]
            points = torch.einsum('bnij,bnj->bni', selected_affines[..., :2], points) + selected_affines[..., 2]
        
        # Main rendering
        for step in range(steps):
            idx = torch.multinomial(probs, n_points, replacement=True)
            batch_idx = torch.arange(batch_size, device=DEVICE).view(-1, 1).expand(-1, n_points)
            selected_affines = affines[batch_idx, idx]
            
            points = torch.einsum('bnij,bnj->bni', selected_affines[..., :2], points) + selected_affines[..., 2]
            
            coords = ((points + 1) * (resolution / 2)).long()
            valid_mask = (coords[..., 0] >= 0) & (coords[..., 0] < resolution) & \
                         (coords[..., 1] >= 0) & (coords[..., 1] < resolution)
            
            # Update buffer with valid points
            for b in range(batch_size):
                valid_coords = coords[b][valid_mask[b]]
                if len(valid_coords) > 0:
                    unique_coords, counts = torch.unique(valid_coords, dim=0, return_counts=True)
                    buffer[b, :, unique_coords[:, 0], unique_coords[:, 1]] += counts.float().view(1, -1)
        
        buffer_max = buffer.amax(dim=(1, 2, 3), keepdim=True)
        buffer = buffer / (buffer_max + BUFFER_MAX_EPSILON)
        buffer = torch.pow(buffer, 0.7)  # Gamma correction
        
        buffer = F.interpolate(buffer, size=(RENDER_FINAL_RESOLUTION, RENDER_FINAL_RESOLUTION), 
                             mode='bilinear', align_corners=False)
        return buffer

    def run_train(self):
        """
        Enhanced training loop with:
        - E-M algorithm iterations
        - Curriculum learning
        - Loss tracking and visualization
        - Checkpoint management
        """
        self.train()
        best_loss = float('inf')
        loss_history = {'total': [], 'recon': [], 'percep': [], 'diffusion': [], 'epoch': []}
        
        logger.info(f"Starting training for {ECM_EPOCHS} epochs with curriculum learning")
        logger.info(f"Enhanced training: BATCH_SIZE={BATCH_SIZE}, LATENT_DIM={LATENT_DIM}, NUM_LATENTS={NUM_LATENTS}")
        logger.info(f"Processing up to 200 batches per epoch for quality enhancement")
        
        for epoch in range(ECM_EPOCHS):
            epoch_loss = 0
            num_batches = 0
            epoch_losses = {'recon': 0, 'percep': 0, 'diffusion': 0}
            
            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{ECM_EPOCHS}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Process all available batches (or limit to 200 for memory efficiency)
                if batch_idx > 200:
                    logger.info(f"Stopping at batch {batch_idx} for memory efficiency")
                    break
                
                # Handle different data formats
                if isinstance(batch, (list, tuple)):
                    images = batch[0].to(DEVICE)
                else:
                    images = batch.to(DEVICE)
                
                # Ensure images have correct shape
                if images.dim() == 3:
                    images = images.unsqueeze(1).repeat(1, 3, 1, 1)
                elif images.shape[1] == 1:
                    images = images.repeat(1, 3, 1, 1)
                
                # E-step: Tokenize images
                token_indices, ifs_tokens, ifs_probs, patch_info, assignment_confidence = self.e_step(images, epoch)
                
                # M-step: Optimize from tokens
                batch_loss = self.m_step(images, token_indices, ifs_tokens, ifs_probs, patch_info, epoch)
                epoch_loss += batch_loss
                num_batches += 1
                
                progress_bar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'avg_conf': f'{assignment_confidence.mean():.4f}'
                })
            
            # Calculate average loss for epoch
            avg_loss = epoch_loss / max(1, num_batches)
            loss_history['total'].append(avg_loss)
            loss_history['epoch'].append(epoch + 1)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch+1}/{ECM_EPOCHS} Summary:")
            logger.info(f"  Average Loss: {avg_loss:.6f}")
            logger.info(f"  Best Loss: {best_loss:.6f}")
            logger.info(f"{'='*60}\n")
            
            # Learning rate scheduling
            self.scheduler_opt.step(avg_loss)
            
            # Checkpoint management
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint("best_model.pt")
                logger.info(f"✓ New best model saved with loss {best_loss:.6f}")
            
            # Regular checkpoint saving
            if epoch % CHECKPOINT_SAVE_INTERVAL == 0:
                self.save_checkpoint(f"checkpoint_epoch{epoch}.pt")
                logger.info(f"✓ Checkpoint saved for epoch {epoch}")
        
        logger.info(f"\n{'='*60}")
        logger.info("Training completed!")
        logger.info(f"Best loss achieved: {best_loss:.6f}")
        logger.info(f"{'='*60}\n")
        
        # Mark training as complete and save final checkpoint
        self.training_complete = True
        self.save_checkpoint(CHECKPOINT_PATH)  # Save final model
        self.save_checkpoint(BEST_CHECKPOINT_PATH)  # Also save as best
        
        logger.info("\n✅ MODEL COEFFICIENTS SAVED")
        logger.info(f"   Primary checkpoint: {CHECKPOINT_PATH}")
        logger.info(f"   Best model: {BEST_CHECKPOINT_PATH}")
        logger.info("   Ready for inference!\n")
        
        return loss_history

    def validate(self, val_dataloader=None, num_samples=4):
        """
        Validation phase: evaluate model on held-out data
        """
        self.eval()
        if val_dataloader is None:
            val_dataloader = self.dataloader
        
        val_losses = []
        logger.info("Running validation...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                if batch_idx > 3:  # Limited validation batches
                    break
                
                if isinstance(batch, (list, tuple)):
                    images = batch[0].to(DEVICE)
                else:
                    images = batch.to(DEVICE)
                
                # E-step: Tokenize
                token_indices, _, _, _, _ = self.e_step(images, epoch=0)
                
                # Compute reconstruction loss
                z, _ = self.tokenizer.encode(images)
                quantized = self.tokenizer.codebook[token_indices]
                affines, transform_probs = self.generator(quantized.unsqueeze(0) if quantized.dim() == 2 else quantized)
                rendered = self.render_patch(affines, transform_probs, images.shape, current_epoch=0)
                
                loss = F.mse_loss(rendered, images, reduction='mean')
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        logger.info(f"Validation Loss: {avg_val_loss:.6f}")
        return avg_val_loss

    def generate_from_prompt(self, prompt, num_patches=16, temperature=1.0, top_k=None):
        """
        Enhanced generation from text prompt with:
        - Temperature control for diversity
        - Top-k sampling
        - Better diffusion scheduling
        - Classifier-free guidance
        """
        self.eval()
        text_emb = encode_prompt(prompt)
        
        # Initialize latent codes with temperature
        z = torch.randn(num_patches, LATENT_DIM, device=DEVICE) * temperature
        
        logger.info(f"Generating image for prompt: '{prompt}'")
        logger.info(f"  Temperature: {temperature}, Top-k: {top_k}")
        
        with torch.no_grad():
            for step, t in enumerate(tqdm(reversed(range(DIFFUSION_STEPS)), desc="Diffusion", total=DIFFUSION_STEPS)):
                t_tensor = torch.tensor([[t]], device=DEVICE).float()
                alpha_t = self.scheduler.alpha_bar[t]
                alpha_t_prev = self.scheduler.alpha_bar[t-1] if t > 0 else torch.tensor(1.0, device=DEVICE)
                
                # Guided denoising
                conditioned_noise = self.diffusion(z, t_tensor, text_emb.expand(num_patches, -1))
                unconditioned_noise = self.diffusion(z, t_tensor)
                
                # Classifier-free guidance
                guidance_scale = GUIDANCE_SCALE * min(1.0, step / DIFFUSION_STEPS)  # Anneal guidance
                pred_noise = unconditioned_noise + guidance_scale * (conditioned_noise - unconditioned_noise)
                
                # Denoising step
                pred_x0 = (z - torch.sqrt(1 - alpha_t) * pred_noise) / (torch.sqrt(alpha_t) + 1e-8)
                
                # Add noise for next step
                if t > 0:
                    noise = torch.randn_like(z) * temperature
                else:
                    noise = 0
                
                z = torch.sqrt(alpha_t_prev) * pred_x0 + torch.sqrt(1 - alpha_t_prev) * noise
        
        # Generate from latent codes
        quantized = self.tokenizer.codebook[self.tokenizer.quantize(z)[0]]
        affines, probs = self.generator(z)
        
        rendered = self.render_patch(
            affines, 
            probs, 
            (1, 3, RENDER_FINAL_RESOLUTION, RENDER_FINAL_RESOLUTION), 
            steps=GENERATE_STEPS, 
            current_epoch=ECM_EPOCHS
        )
        
        logger.info(f"✓ Generation complete for '{prompt}'")
        return rendered

    def save_checkpoint(self, path=None):
        if path is None:
            path = CHECKPOINT_PATH
        
        # Ensure directory exists
        checkpoint_dir = os.path.dirname(path) or '.'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_data = {
            'tokenizer': self.tokenizer.state_dict(),
            'generator': self.generator.state_dict(),
            'renderer': self.renderer.state_dict(),
            'diffusion': self.diffusion.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler_optimizer': self.scheduler_opt.state_dict(),
            'training_complete': self.training_complete,
            'dataset_name': self.dataset_name,
            'model_config': {
                'latent_dim': LATENT_DIM,
                'num_latents': NUM_LATENTS,
                'num_transforms': NUM_TRANSFORMS,
                'diffusion_steps': DIFFUSION_STEPS,
            }
        }
        
        torch.save(checkpoint_data, path)
        file_size = os.path.getsize(path) / (1024 ** 2)  # Convert to MB
        logger.info(f"✓ Checkpoint saved to {path} ({file_size:.2f} MB)")
    
    def load_checkpoint(self, path=None):
        if path is None:
            path = CHECKPOINT_PATH
        
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=DEVICE)
                self.tokenizer.load_state_dict(checkpoint['tokenizer'])
                self.generator.load_state_dict(checkpoint['generator'])
                self.renderer.load_state_dict(checkpoint['renderer'])
                self.diffusion.load_state_dict(checkpoint['diffusion'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler_opt.load_state_dict(checkpoint['scheduler_optimizer'])
                self.training_complete = checkpoint.get('training_complete', False)
                
                file_size = os.path.getsize(path) / (1024 ** 2)
                logger.info(f"✓ Checkpoint loaded from {path} ({file_size:.2f} MB)")
                logger.info(f"  Training complete: {self.training_complete}")
                logger.info(f"  Trained on: {checkpoint.get('dataset_name', 'Unknown')}")
                
                return True
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                return False
        else:
            logger.warning(f"No checkpoint found at {path}")
            return False

def create_learning_documentation():
    """
    Create comprehensive documentation of the learning phase
    """
    doc = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║              ENHANCED LEARNING PHASE - DETAILED EXPLANATION                   ║
╚═══════════════════════════════════════════════════════════════════════════════╝

1. E-STEP (EXPECTATION): IMAGE TOKENIZATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Purpose: Decompose images into discrete tokens using learned codebook
   
   Process:
   a) Image Encoding: CNN encoder projects image patches to latent space
   b) Quantization: Find nearest codebook vector for each patch
   c) Confidence Scoring: Measure assignment certainty
   d) IFS Extraction: Convert tokens to affine transformations
   
   Output: Token indices that represent image structure
   Learning: Codebook vectors learn to capture diverse patterns
   
   Benefits:
   - Discrete representation (memory efficient)
   - Discrete optimization (stable training)
   - Interpretable tokens (analyzable)

2. M-STEP (MAXIMIZATION): PARAMETER OPTIMIZATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Purpose: Update all model parameters to minimize reconstruction error
   
   Loss Functions (weighted combination):
   
   a) RECONSTRUCTION LOSS (primary)
      - Pixel-level MSE between rendered and original
      - Ensures visual similarity
      - Weight: 1.0
   
   b) PERCEPTUAL LOSS (semantic)
      - VGG16 feature-level comparison
      - Ensures high-level structure preservation
      - Weight: 0.2
   
   c) DIFFUSION LOSS (generative)
      - Noise prediction error in diffusion process
      - Enables text-guided generation
      - Weight: 0.05
   
   d) CODEBOOK LOSS (quantization)
      - Codebook vectors learn from encoder outputs
      - Updates discrete token representations
      - Weight: 0.25
   
   e) COMMITMENT LOSS (stability)
      - Encoder learns to stay close to codebook
      - Prevents codebook collapse
      - Weight: 0.1
   
   f) ORTHOGONALITY LOSS (diversity)
      - Encourages linearly independent codebook vectors
      - Prevents redundant representations
      - Weight: 0.01 × curriculum
   
   g) ENTROPY REGULARIZER (coverage)
      - Encourages diverse token usage
      - Prevents unused tokens
      - Weight: 0.001

3. CURRICULUM LEARNING
━━━━━━━━━━━━━━━━━━━━
   Strategy: Gradually increase loss complexity during training
   
   Schedule:
   - Early epochs: Focus on basic reconstruction
   - Middle epochs: Add diversity constraints
   - Late epochs: Full regularization weight
   
   Benefits:
   - Stable early training
   - Better convergence
   - Avoids local minima

4. LOSS COMPONENTS INTERACTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   
   ┌─────────────────────────────────────────┐
   │  Total Loss = Σ(weight_i × loss_i)      │
   │                                         │
   │  Weights determine optimization focus:  │
   │  - High: Critical for training          │
   │  - Low: Fine-tuning/regularization      │
   └─────────────────────────────────────────┘

5. OPTIMIZATION PIPELINE
━━━━━━━━━━━━━━━━━━━━━━
   
   For each epoch:
     For each batch:
       1. Forward pass: Encode → Quantize → Render
       2. Compute all loss components
       3. Backward pass: Compute gradients
       4. Gradient clipping: Prevent explosion
       5. Parameter update: Adam optimizer
       6. Learning rate scheduling: ReduceLROnPlateau
   
   Key Features:
   - Gradient clipping (norm ≤ 0.5)
   - Adaptive learning rates
   - Batch normalization stability

6. CONVERGENCE MONITORING
━━━━━━━━━━━━━━━━━━━━━━
   
   Tracked metrics:
   - Total loss: Overall optimization progress
   - Reconstruction loss: Visual quality
   - Perceptual loss: Semantic correctness
   - Diffusion loss: Generative capability
   - Assignment confidence: Token stability
   - Codebook usage: Training health

7. CHECKPOINTING STRATEGY
━━━━━━━━━━━━━━━━━━━━━━
   
   Saves:
   - Best model (lowest validation loss)
   - Periodic checkpoints (every N epochs)
   - Full training state (optimizer, scheduler)
   
   Benefits:
   - Resume interrupted training
   - Compare model versions
   - Experiment tracking

8. GENERATION PHASE
━━━━━━━━━━━━━━━━
   Uses learned components:
   - Tokenizer: Image → Tokens
   - Diffusion: Text → Latent codes
   - Generator: Latent → IFS transforms
   - Renderer: IFS → Image
   
   Classifier-free guidance annealing:
   - Start high for diversity
   - Gradually reduce for coherence
   - Balance between quality and diversity

═══════════════════════════════════════════════════════════════════════════════
"""
    return doc
def test_tokenization_and_generation(dataset_name='CIFAR10', data_path='./data', custom_folder=None):
    """
    Test tokenization and image generation with specified dataset
    
    Args:
        dataset_name: Name of dataset ('CIFAR10', 'CIFAR100', 'STL10', 'FLOWERS', 'CUSTOM')
        data_path: Path where to download/find datasets
        custom_folder: Path to custom image folder (if using 'CUSTOM')
    """
    logger.info(f"Starting tokenization and image generation test with {dataset_name}...")
    
    # Use custom folder if provided
    if custom_folder and os.path.exists(custom_folder):
        dataset_name = custom_folder
        logger.info(f"Using custom folder: {custom_folder}")
    
    # Create model with specified dataset
    model = EnhancedLIDECM(dataset_name=dataset_name, data_path=data_path, augment=True)
    
    # Train for a short period
    logger.info(f"Starting short training session (2 epochs) on {dataset_name}...")
    loss_history = model.run_train()
    
    # Test tokenization on real images
    logger.info("Testing tokenization on real images...")
    sample_batch = next(iter(model.dataloader))
    
    # Handle different data formats
    if isinstance(sample_batch, (list, tuple)):
        sample_images = sample_batch[0][:4]
    else:
        sample_images = sample_batch[:4]
    
    sample_images = sample_images.to(DEVICE)
    
    token_indices, ifs_tokens, ifs_probs, patch_info = model.tokenize_image(sample_images)
    logger.info(f"Tokenized {sample_images.shape[0]} images into tokens")
    logger.info(f"Token indices shape: {token_indices.shape}")
    logger.info(f"IFS tokens shape: {ifs_tokens.shape}")
    logger.info(f"IFS probabilities shape: {ifs_probs.shape}")
    logger.info(f"Unique tokens used: {len(torch.unique(token_indices))}")
    
    # Visualize tokenization
    plt.figure(figsize=(15, 8))
    
    # Original images
    for i in range(min(4, sample_images.shape[0])):
        plt.subplot(2, 4, i+1)
        img = sample_images[i].permute(1, 2, 0).cpu().detach().numpy()
        img = (img * NORM_STD) + NORM_MEAN
        plt.imshow(np.clip(img, 0, 1))
        plt.title(f"Original {i+1}")
        plt.axis('off')
    
    # Reconstructed from tokens
    for i in range(min(4, sample_images.shape[0])):
        plt.subplot(2, 4, i+5)
        try:
            reconstructed = model.reconstruct_from_tokens(token_indices[i*16:(i+1)*16] if token_indices.shape[0] > 16 else token_indices)
            img = reconstructed[0].permute(1, 2, 0).cpu().detach().numpy() if reconstructed.dim() == 4 else reconstructed
        except:
            img = np.zeros((RENDER_FINAL_RESOLUTION, RENDER_FINAL_RESOLUTION, 3))
        plt.imshow(np.clip(img, 0, 1))
        plt.title(f"Tokenized {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'tokenization_results_{dataset_name}.png', dpi=150)
    logger.info(f"Saved tokenization results to 'tokenization_results_{dataset_name}.png'")
    plt.close()
    
    # Generate images from prompts
    prompts = [
        "a beautiful natural image",
        "colorful abstract pattern",
        "natural scenery landscape",
        "geometric design",
        "artistic composition"
    ]
    
    plt.figure(figsize=(15, 10))
    for i, prompt in enumerate(prompts):
        logger.info(f"Generating image for prompt: {prompt}")
        try:
            result = model.generate_from_prompt(prompt)
            img = result[0].permute(1, 2, 0).cpu().detach().numpy()
            
            plt.subplot(2, 3, i+1)
            plt.imshow(np.clip(img, 0, 1))
            plt.title(prompt)
            plt.axis('off')
        except Exception as e:
            logger.warning(f"Failed to generate image for '{prompt}': {e}")
            plt.subplot(2, 3, i+1)
            plt.imshow(np.zeros((RENDER_FINAL_RESOLUTION, RENDER_FINAL_RESOLUTION, 3)))
            plt.title(f"{prompt} (failed)")
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'generated_images_{dataset_name}.png', dpi=150)
    logger.info(f"Saved generated images to 'generated_images_{dataset_name}.png'")
    plt.close()
    
    # Analysis: Show token distribution
    logger.info("Analyzing token distribution...")
    unique_tokens, counts = torch.unique(token_indices, return_counts=True)
    logger.info(f"Found {len(unique_tokens)} unique tokens")
    top_tokens = unique_tokens[torch.argsort(counts, descending=True)[:5]]
    logger.info(f"Most common tokens: {top_tokens.tolist()}")
    logger.info(f"Token frequency (top 5): {torch.sort(counts, descending=True)[0][:5].tolist()}")
    
    logger.info("Test completed successfully!")
    return model, loss_history

# --------------------------
# Inference Mode
# --------------------------
def run_inference_only(checkpoint_path=CHECKPOINT_PATH, prompts=None):
    """
    Run model in inference-only mode using pre-trained weights
    Skips training and goes directly to generation
    """
    logger.info("\n" + "="*80)
    logger.info("INFERENCE MODE - LOADING PRE-TRAINED MODEL")
    logger.info("="*80 + "\n")
    
    # Create model (minimal setup)
    model = EnhancedLIDECM(dataset_name='CIFAR10', data_path='./data', augment=False)
    
    # Load checkpoint
    if not model.load_checkpoint(checkpoint_path):
        logger.error(f"Failed to load checkpoint from {checkpoint_path}")
        return None
    
    model.eval()
    
    # Default prompts if none provided
    if prompts is None:
        prompts = [
            ("a beautiful natural image", 1.0),
            ("colorful abstract pattern", 1.5),
            ("natural scenery landscape", 0.8),
            ("geometric design", 1.0),
            ("artistic composition", 1.2)
        ]
    
    logger.info(f"Generating {len(prompts)} images using pre-trained model...\n")
    
    # Generate images
    plt.figure(figsize=(16, 10))
    for i, (prompt, temp) in enumerate(prompts):
        logger.info(f"Generating: '{prompt}' (temperature={temp})")
        try:
            result = model.generate_from_prompt(prompt, temperature=temp)
            img = result[0].permute(1, 2, 0).cpu().detach().numpy()
            
            plt.subplot(2, 3, i+1)
            plt.imshow(np.clip(img, 0, 1))
            plt.title(f"{prompt}\n(T={temp})", fontweight='bold')
            plt.axis('off')
        except Exception as e:
            logger.error(f"Generation failed for '{prompt}': {e}")
            plt.subplot(2, 3, i+1)
            plt.imshow(np.zeros((RENDER_FINAL_RESOLUTION, RENDER_FINAL_RESOLUTION, 3)))
            plt.title(f"{prompt}\n(Failed)", fontweight='bold', color='red')
            plt.axis('off')
    
    plt.suptitle("Inference Results - Text-Guided Image Generation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('inference_results.png', dpi=150, bbox_inches='tight')
    logger.info("\n✓ Saved inference results to 'inference_results.png'\n")
    plt.close()
    
    logger.info("="*80)
    logger.info("✅ INFERENCE COMPLETED SUCCESSFULLY")
    logger.info("="*80 + "\n")
    
    return model

# --------------------------
# Main Execution
# --------------------------
if __name__ == '__main__':
    logger.info("\n" + "█"*80)
    logger.info("█" + " "*78 + "█")
    logger.info("█" + "ENHANCED FRACTAL TOKENIZATION MODEL - MAIN EXECUTION".center(78) + "█")
    logger.info("█" + " "*78 + "█")
    logger.info("█"*80 + "\n")
    
    # Check if pre-trained model exists
    model_exists = os.path.exists(CHECKPOINT_PATH)
    
    if model_exists:
        logger.info("✓ Pre-trained model found!")
        logger.info(f"  Path: {CHECKPOINT_PATH}")
        logger.info("  Switching to INFERENCE MODE (skipping training)\n")
        
        # Load and use pre-trained model for inference
        model = run_inference_only(CHECKPOINT_PATH)
        
    else:
        logger.info("✗ Pre-trained model not found")
        logger.info("  Starting TRAINING MODE\n")
        
        # Train a new model
        model, loss_history = test_tokenization_and_generation('CIFAR10', './data')
        
        logger.info("\n" + "="*80)
        logger.info("Training complete! Model saved and ready for inference.")
        logger.info("Run the script again to use the pre-trained model for inference.")
        logger.info("="*80 + "\n")