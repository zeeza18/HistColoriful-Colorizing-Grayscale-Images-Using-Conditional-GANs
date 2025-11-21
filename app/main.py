# ============================================================================
# main.py - FastAPI Image Colorization Application
# ============================================================================

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from PIL import Image
import io
import base64
from pathlib import Path
import time
from skimage.color import rgb2lab, lab2rgb
from torchvision import transforms
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter
from skimage import exposure
from typing import Optional

try:
    from .models import load_colorization_model
except ImportError:
    from models import load_colorization_model

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
STATIC_DIR = BASE_DIR / "static"
CHECKPOINT_DIR = PROJECT_ROOT / "model_checkpoints"
GAN_MODEL_PATH = CHECKPOINT_DIR / "ablation_lambda50.pt"

app = FastAPI(
    title="Image Colorization API",
    description="Classical and Deep Learning Image Colorization",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Path("uploads").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# ============================================================================
# LOAD MODELS
# ============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    gan_model = load_colorization_model(str(GAN_MODEL_PATH), device=device)
    print("✓ GAN model loaded successfully")
except Exception as e:
    print(f"Warning: Could not load GAN model: {e}")
    gan_model = None

# ============================================================================
# COLORIZATION FUNCTIONS
# ============================================================================

def classical_histogram_matching(gray_img, reference_img):
    gray_lab = rgb2lab(gray_img)
    ref_lab = rgb2lab(reference_img)
    
    matched_lab = gray_lab.copy()
    matched_lab[:, :, 1] = exposure.match_histograms(gray_lab[:, :, 1], ref_lab[:, :, 1], channel_axis=None)
    matched_lab[:, :, 2] = exposure.match_histograms(gray_lab[:, :, 2], ref_lab[:, :, 2], channel_axis=None)
    
    colorized_rgb = lab2rgb(matched_lab)
    return colorized_rgb


def classical_kmeans_colorization(gray_img, reference_img, n_clusters=8):
    gray_lab = rgb2lab(gray_img)
    ref_lab = rgb2lab(reference_img)
    
    gray_L = gray_lab[:, :, 0].flatten().reshape(-1, 1)
    ref_L = ref_lab[:, :, 0].flatten()
    ref_ab = ref_lab[:, :, 1:].reshape(-1, 2)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    ref_clusters = kmeans.fit_predict(ref_L.reshape(-1, 1))
    
    cluster_colors = np.zeros((n_clusters, 2))
    for i in range(n_clusters):
        cluster_mask = (ref_clusters == i)
        if cluster_mask.sum() > 0:
            cluster_colors[i] = ref_ab[cluster_mask].mean(axis=0)
    
    gray_clusters = kmeans.predict(gray_L)
    
    colorized_lab = gray_lab.copy()
    colorized_ab = cluster_colors[gray_clusters].reshape(gray_lab.shape[0], gray_lab.shape[1], 2)
    colorized_lab[:, :, 1:] = colorized_ab
    
    colorized_rgb = lab2rgb(colorized_lab)
    return colorized_rgb


def classical_gaussian_local_colorization(gray_img, reference_img, sigma=5, window_size=16):
    gray_lab = rgb2lab(gray_img)
    ref_lab = rgb2lab(reference_img)
    
    gray_L_smooth = gaussian_filter(gray_lab[:, :, 0], sigma=sigma)
    ref_L_smooth = gaussian_filter(ref_lab[:, :, 0], sigma=sigma)
    
    colorized_lab = gray_lab.copy()
    h, w = gray_lab.shape[:2]
    half_window = window_size // 2
    
    for i in range(0, h, window_size):
        for j in range(0, w, window_size):
            i_start = max(0, i - half_window)
            i_end = min(h, i + half_window)
            j_start = max(0, j - half_window)
            j_end = min(w, j + half_window)
            
            gray_local_L = gray_L_smooth[i_start:i_end, j_start:j_end].flatten()
            ref_local_L = ref_L_smooth[i_start:i_end, j_start:j_end].flatten()
            ref_local_ab = ref_lab[i_start:i_end, j_start:j_end, 1:].reshape(-1, 2)
            
            if len(gray_local_L) == 0 or len(ref_local_L) == 0:
                continue
            
            mean_gray_L = np.mean(gray_local_L)
            luminosity_diff = np.abs(ref_local_L - mean_gray_L)
            similar_mask = luminosity_diff < 15
            
            if similar_mask.sum() > 0:
                mean_ab = ref_local_ab[similar_mask].mean(axis=0)
            else:
                mean_ab = ref_local_ab.mean(axis=0)
            
            colorized_lab[i:i+window_size, j:j+window_size, 1:] = mean_ab
    
    colorized_rgb = lab2rgb(colorized_lab)
    return colorized_rgb


def gan_colorization(gray_img):
    if gan_model is None:
        raise HTTPException(status_code=500, detail="GAN model not loaded")
    
    gray_rgb = np.stack([gray_img, gray_img, gray_img], axis=-1)
    lab = rgb2lab(gray_rgb).astype("float32")
    lab_tensor = transforms.ToTensor()(lab)
    L = lab_tensor[[0], ...] / 50.0 - 1.0
    L_input = L.unsqueeze(0).to(device)
    
    with torch.no_grad():
        gen_ab = gan_model.generator(L_input)
    
    L_denorm = (L_input + 1.) * 50
    ab_denorm = gen_ab * 128
    Lab = torch.cat([L_denorm, ab_denorm], dim=1).permute(0, 2, 3, 1).cpu().numpy()[0]
    
    colorized_rgb = lab2rgb(Lab)
    return colorized_rgb


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def image_to_base64(image_array):
    image_array = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(image_array)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def load_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize((256, 256))
    return np.array(img) / 255.0


def grayscale_image(img_array):
    gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
    return gray


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def home():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return "<h1>Image Colorization API</h1><p>Upload index.html to static folder</p>"
    with open(index_file, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/info")
async def get_info():
    return {
        "title": "Image Colorization API",
        "version": "1.0.0",
        "methods": [
            {"id": "histogram", "name": "Histogram Matching", "type": "Classical"},
            {"id": "kmeans", "name": "K-means Color Transfer", "type": "Classical"},
            {"id": "gaussian", "name": "Gaussian + Local", "type": "Classical"},
            {"id": "gan", "name": "GAN (Lambda=50)", "type": "Deep Learning"}
        ],
        "device": str(device),
        "gan_available": gan_model is not None
    }


@app.post("/api/colorize")
async def colorize_image(
    method: str = Form(...),
    grayscale_file: UploadFile = File(...),
    reference_image: Optional[UploadFile] = File(None)
):
    try:
        start_time = time.time()
        
        gray_bytes = await grayscale_file.read()
        gray_img = load_image(gray_bytes)
        gray_single = grayscale_image(gray_img)
        
        if method in ["histogram", "kmeans", "gaussian"]:
            if reference_image is None:
                raise HTTPException(status_code=400, detail="Reference image required for classical methods")
            
            ref_bytes = await reference_image.read()
            ref_img = load_image(ref_bytes)
            gray_3ch = np.stack([gray_single, gray_single, gray_single], axis=-1)
            
            if method == "histogram":
                result = classical_histogram_matching(gray_3ch, ref_img)
            elif method == "kmeans":
                result = classical_kmeans_colorization(gray_3ch, ref_img)
            elif method == "gaussian":
                result = classical_gaussian_local_colorization(gray_3ch, ref_img)
                
        elif method == "gan":
            result = gan_colorization(gray_single)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {method}")
        
        processing_time = time.time() - start_time
        
        result_base64 = image_to_base64(result)
        
        return JSONResponse({
            "success": True,
            "method": method,
            "processing_time": round(processing_time, 3),
            "colorized_image": result_base64
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": str(device),
        "gan_loaded": gan_model is not None
    }


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("=" * 80)
    print("IMAGE COLORIZATION API")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"GAN Model: {'Loaded' if gan_model else 'Not Available'}")
    print("\nStarting server...")
    print("Access at: http://localhost:8000")
    print("=" * 80)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
