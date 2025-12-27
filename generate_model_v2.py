# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Generate 3D model from a folder of images using VGGT.

Usage:
    python generate_model_v2.py <model_id> [--name "Display Name"] [--conf-thres 50.0]
    
Examples:
    python generate_model_v2.py grove
    python generate_model_v2.py grove --name "The Grove"
    python generate_model_v2.py town_track --conf-thres 30.0
    
The script expects images to be in: images/3d/<model_id>/
The GLB file will be saved to: images/3d/<model_id>.glb
"""

import os
import sys
import torch
import numpy as np
import glob
import gc
import argparse
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VGGT_DIR = os.path.join(SCRIPT_DIR, "thirdparty", "vggt")
if VGGT_DIR not in sys.path:
    sys.path.insert(0, VGGT_DIR)

# Path to images/3d/ folder (relative to this script's location)
IMAGES_3D_DIR = os.path.join(SCRIPT_DIR, "..", "images", "3d")
UPDATE_MODELS_SCRIPT = os.path.join(IMAGES_3D_DIR, "update_models_json.py")

from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


def load_model():
    """Load and initialize the VGGT model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Initializing and loading VGGT model...")
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    
    model.eval()
    model = model.to(device)
    
    return model, device


def run_model(image_folder: str, model, device: str) -> dict:
    """
    Run the VGGT model on images in the specified folder and return predictions.
    
    Args:
        image_folder: Path to folder containing images
        model: Loaded VGGT model
        device: Device to run on ('cuda' or 'cpu')
    
    Returns:
        Dictionary of predictions
    """
    print(f"Processing images from {image_folder}")

    if device == "cpu":
        print("Warning: Running on CPU. This will be slow. CUDA recommended.")

    # Load and preprocess images
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    image_names = []
    for ext in image_extensions:
        image_names.extend(glob.glob(os.path.join(image_folder, ext)))
    image_names = sorted(image_names)
    
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError(f"No images found in {image_folder}")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    # Run inference
    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
        else:
            predictions = model(images)

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
    predictions['pose_enc_list'] = None  # remove pose_enc_list

    # Generate world points from depth map
    print("Computing world points from depth map...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    # Clean up
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return predictions


def generate_glb(
    model_id: str,
    name: str = None,
    conf_thres: float = 50.0,
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    show_cam: bool = True,
    mask_sky: bool = False,
    prediction_mode: str = "Depthmap and Camera Branch",
):
    """
    Generate a GLB file from images in the model folder.
    
    Args:
        model_id: The model ID (folder name in images/3d/)
        name: Display name for the model (used when updating models.json)
        conf_thres: Confidence threshold for filtering points (0-100)
        mask_black_bg: Remove black background points
        mask_white_bg: Remove white background points
        show_cam: Include camera positions in the model
        mask_sky: Remove sky points
        prediction_mode: "Depthmap and Camera Branch" or "Pointmap Branch"
    
    Returns:
        Path to the generated GLB file
    """
    # Resolve paths
    image_folder = os.path.abspath(os.path.join(IMAGES_3D_DIR, model_id))
    output_glb = os.path.abspath(os.path.join(IMAGES_3D_DIR, f"{model_id}.glb"))
    
    if not os.path.isdir(image_folder):
        raise ValueError(f"Image folder not found: {image_folder}")
    
    print(f"\n{'='*60}")
    print(f"Generating 3D model for: {model_id}")
    print(f"{'='*60}")
    print(f"Image folder: {image_folder}")
    print(f"Output GLB: {output_glb}")
    print(f"Confidence threshold: {conf_thres}%")
    print(f"Prediction mode: {prediction_mode}")
    print()
    
    # Load model
    model, device = load_model()
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Run model on images
    print("\nRunning VGGT model...")
    with torch.no_grad():
        predictions = run_model(image_folder, model, device)
    
    # Save predictions (useful for debugging/re-visualization)
    predictions_path = os.path.join(image_folder, "predictions.npz")
    np.savez(predictions_path, **predictions)
    print(f"Saved predictions to: {predictions_path}")
    
    # Create a temporary directory structure for visual_util
    # (it expects target_dir/images structure)
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    temp_images_dir = os.path.join(temp_dir, "images")
    os.makedirs(temp_images_dir)
    
    # Copy images to temp directory
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    for ext in image_extensions:
        for img_path in glob.glob(os.path.join(image_folder, ext)):
            shutil.copy(img_path, temp_images_dir)
    
    # Convert predictions to GLB
    print("\nConverting to GLB format...")
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames="All",
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        mask_sky=mask_sky,
        target_dir=temp_dir,
        prediction_mode=prediction_mode,
    )
    
    # Export GLB
    glbscene.export(file_obj=output_glb)
    print(f"\nSaved GLB file to: {output_glb}")
    
    # Cleanup temp directory
    shutil.rmtree(temp_dir)
    
    # Cleanup
    del predictions
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return output_glb


def run_update_models_json(model_id: str, name: str = None):
    """
    Run update_models_json.py to update models.json with the new model.
    
    Args:
        model_id: The model ID
        name: Display name for the model (optional)
    """
    if not os.path.exists(UPDATE_MODELS_SCRIPT):
        print(f"Warning: update_models_json.py not found at {UPDATE_MODELS_SCRIPT}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Updating models.json...")
    print(f"{'='*60}")
    
    cmd = [sys.executable, UPDATE_MODELS_SCRIPT, model_id]
    if name:
        cmd.extend(["--name", name])
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=IMAGES_3D_DIR)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running update_models_json.py: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D GLB model from images using VGGT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_model_v2.py grove
    python generate_model_v2.py grove --name "The Grove"
    python generate_model_v2.py town_track --conf-thres 30.0
    
The script expects images to be in: images/3d/<model_id>/
The GLB file will be saved to: images/3d/<model_id>.glb
        """
    )
    parser.add_argument(
        "model_id",
        help="The model ID (should match the folder name in images/3d/ containing images)"
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Display name for the model (auto-generated from ID if not provided)"
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=50.0,
        help="Confidence threshold for filtering points, 0-100 (default: 50.0)"
    )
    parser.add_argument(
        "--mask-black-bg",
        action="store_true",
        help="Remove black background points"
    )
    parser.add_argument(
        "--mask-white-bg",
        action="store_true",
        help="Remove white background points"
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Don't include camera positions in the model"
    )
    parser.add_argument(
        "--mask-sky",
        action="store_true",
        help="Remove sky points"
    )
    parser.add_argument(
        "--prediction-mode",
        choices=["Depthmap and Camera Branch", "Pointmap Branch"],
        default="Depthmap and Camera Branch",
        help="Prediction mode (default: 'Depthmap and Camera Branch')"
    )
    parser.add_argument(
        "--skip-update-json",
        action="store_true",
        help="Skip running update_models_json.py after generating the model"
    )
    
    args = parser.parse_args()
    
    try:
        # Generate the GLB file
        output_glb = generate_glb(
            model_id=args.model_id,
            name=args.name,
            conf_thres=args.conf_thres,
            mask_black_bg=args.mask_black_bg,
            mask_white_bg=args.mask_white_bg,
            show_cam=not args.no_camera,
            mask_sky=args.mask_sky,
            prediction_mode=args.prediction_mode,
        )
        
        print(f"\n✓ Successfully generated: {output_glb}")
        
        # Update models.json
        if not args.skip_update_json:
            success = run_update_models_json(args.model_id, args.name)
            if success:
                print(f"\n✓ Successfully updated models.json")
            else:
                print(f"\n✗ Failed to update models.json")
                sys.exit(1)
        
        print(f"\n{'='*60}")
        print(f"Done! Model '{args.model_id}' is ready.")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
