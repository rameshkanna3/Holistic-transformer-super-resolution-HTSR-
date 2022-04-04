import os
import cv2
import torch
import streamlit as st
import numpy as np

@st.cache(persist=True,allow_output_mutation=True,show_spinner=False)
def loadModel(device: str):
    is_gpu_available = torch.cuda.is_available()
    if device == "GPU" and is_gpu_available:
        model = torch.load("pretrained/htsr_gpu.pth")
        device = torch.device("cuda")
    else:
        model = torch.load("pretrained/htsr_cpu.pth")
        device = torch.device("cpu")
    return (model, device)

@st.cache(persist=True,allow_output_mutation=True,show_spinner=False)
def transform(impath: str, device: torch.tensor, window_size: int = 8):
    img_name, _ = os.path.splitext(os.path.basename(impath))
    img_bgr = cv2.imread(impath, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img_rgb = np.transpose(img_bgr if img_bgr.shape[2] == 1 else img_bgr[:, :, [2, 1, 0]], (2, 0, 1))
    image = torch.from_numpy(img_rgb).float().unsqueeze(0).to(device)
    _, _, height, width = image.size()
    height_pad = (height // window_size + 1) * window_size - height
    width_pad = (width // window_size + 1) * window_size - width
    image = torch.cat([image, torch.flip(image, [2])], 2)[:, :, :height + height_pad, :]
    image = torch.cat([image, torch.flip(image, [3])], 3)[:, :, :, :width + width_pad]
    return (image, img_name)

def predict(image: torch.tensor, img_name: str, model: object, scale: int, tile: int = 640):
    with st.spinner("Generating super res image..."):
        with torch.no_grad():
            b, c, h, w = image.size()
            tile = min(tile, h, w)
            tile_overlap = 32
            stride = tile - tile_overlap
            height_idxs = list(range(0, h-tile, stride)) + [h-tile]
            width_idxs = list(range(0, w-tile, stride)) + [w-tile]
            E = torch.zeros(b, c, h*scale, w*scale).type_as(image)
            W = torch.zeros_like(E)

            for h_idx in height_idxs:
                for w_idx in width_idxs:
                    in_patch = image[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                    out_patch = model(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)
                    E[..., h_idx*scale:(h_idx+tile)*scale, w_idx*scale:(w_idx+tile)*scale].add_(out_patch)
                    W[..., h_idx*scale:(h_idx+tile)*scale, w_idx*scale:(w_idx+tile)*scale].add_(out_patch_mask)
            sr_image = E.div_(W)
            sr_image = sr_image.data.squeeze().float().cpu().clamp_(0,1).numpy()
            if sr_image.ndim == 3:
                sr_image = np.transpose(sr_image[[2, 1, 0], :, :], (1, 2, 0))
            sr_image = (sr_image * 255.0).round().astype(np.uint8)
            cv2.imwrite(f"sr_out/{img_name}_HTSR.png", sr_image)
