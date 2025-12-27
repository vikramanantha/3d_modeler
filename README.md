# 3D Modeler

3D Model her? I hardly know her!

**Vikram Anantha**  
**Dec 2025**

## Background

I have a lot of Aerial Photos. It would be cool to make 3D models with the photos that I have, especially since most of the aerial shoots I have would do this.

## The Project

Use VGGT to process the photos that I have, and then generate the model. Display the Ply on the website.

## Run (Gradio UI)

From repo root:

```bash
cd /Users/markivanantha/Documents/vikramanantha.github.io

# 1) Make sure your env uses numpy<2 (important on Python 3.12)
pip install -r 3d_modeler/requirements_vggt_ui.txt

# 2) Install torch/torchvision for your platform (CPU or CUDA)
# (torch install varies by OS/GPU, so follow pytorch.org instructions)

# 3) Launch UI
python3 3d_modeler/vggt_photos_to_glb.py --ui
```

## Run (CLI)

```bash
cd /Users/markivanantha/Documents/vikramanantha.github.io/3d_modeler
python3 vggt_photos_to_glb.py --photos_dir /path/to/photos --out ../images/3d
```

If `--out` is a directory, the script will write `vggt_output.glb` inside it.

**This script now generates the `.glb` directly (no COLMAP export).**
