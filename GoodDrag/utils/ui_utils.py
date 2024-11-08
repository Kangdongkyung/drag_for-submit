# *************************************************************************
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *************************************************************************


import os
import shutil
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import gradio as gr
from copy import deepcopy
from einops import rearrange
from types import SimpleNamespace

import datetime
import PIL
from PIL import Image
from PIL.ImageOps import exif_transpose
import torch
import torch.nn.functional as F

from diffusers import DDIMScheduler, AutoencoderKL
from pipeline import GoodDragger

from torchvision.utils import save_image
from pytorch_lightning import seed_everything

from .lora_utils import train_lora



## UniPose 사용하기 위해서 import함.
import re
import sys
import matplotlib.pyplot as plt
import argparse

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the path to 'X-Pose' directory to sys.path
xpose_dir = os.path.join(script_dir, '..', '..','X-Pose')
if xpose_dir not in sys.path:
    sys.path.append(xpose_dir)
    
# Add the app directory explicitly to sys.path
app_dir = os.path.join(xpose_dir)
if app_dir not in sys.path:
    sys.path.append(app_dir)
    

# Load predefined keypoints
from predefined_keypoints import *
from app import load_image, get_unipose_output, plot_on_image, load_model_for_gooddrag


# -------------- general UI functionality --------------
def clear_all(length=512):
    return gr.Image.update(value=None, height=length, width=length), \
           gr.Image.update(value=None, height=length, width=length), \
           gr.Image.update(value=None, height=length, width=length), \
           [], None, None


def mask_image(image,
               mask,
               color=[255, 0, 0],
               alpha=0.5):
    """ Overlay mask on image for visualization purpose.
    Args:
        image (H, W, 3) or (H, W): input image
        mask (H, W): mask to be overlaid
        color: the color of overlaid mask
        alpha: the transparency of the mask
    """
    out = deepcopy(image)
    img = deepcopy(image)
    img[mask == 1] = color
    out = cv2.addWeighted(img, alpha, out, 1 - alpha, 0, out)
    return out


def store_img(img, length=512):
    image, mask = img["image"], np.float32(img["mask"][:, :, 0]) / 255.
    height, width, _ = image.shape
    image = Image.fromarray(image)
    image = exif_transpose(image)
    image = image.resize((length, int(length * height / width)), PIL.Image.BILINEAR)
    mask = cv2.resize(mask, (length, int(length * height / width)), interpolation=cv2.INTER_NEAREST)
    image = np.array(image)

    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        masked_img = mask_image(image, 1 - mask, color=[0, 0, 0], alpha=0.3)
    else:
        masked_img = image.copy()
    # when new image is uploaded, `selected_points` should be empty
    return image, [], masked_img, mask


## 아래부터는 UniPose를 위해 추가한 함수들 #########################
# Define the label to index mapping
label_to_instance = {
    "person", 
    "face", 
    "hand", 
    "animal_in_AnimalKindom", 
    "animal_in_AP10K", 
    "animal", 
    "animal_face", 
    "fly", 
    "locust", 
    "car", 
    "short_sleeved_shirt", 
    "long_sleeved_outwear", 
    "short_sleeved_outwear", 
    "sling", 
    "vest", 
    "long_sleeved_dress", 
    "long_sleeved_shirt", 
    "trousers", 
    "sling_dress", 
    "vest_dress", 
    "skirt", 
    "short_sleeved_dress", 
    "shorts", 
    "table", 
    "chair", 
    "bed", 
    "sofa", 
    "swivelchair"
}
# Define the direction to coordinate change mapping
direction_to_offset = {
    "up": (0, -100),
    "down": (0, 100),
    "left": (-75, 0),
    "right": (100, 0),
    "forward": (-100, 100),
}

def parse_input_text(input_text):
    # Find all keypoints and the direction in the input text
    label_pattern = re.compile(r'\b(?:' + '|'.join(re.escape(label) for label in label_to_instance) + r')\b', re.IGNORECASE)
    found_labels = label_pattern.findall(input_text)
    
    direction_pattern = re.compile(r'\b(?:' + '|'.join(re.escape(direction) for direction in direction_to_offset.keys()) + r')\b', re.IGNORECASE)
    found_directions = direction_pattern.findall(input_text)
    
    # Assume the direction is the last part of the input text
    direction = found_directions[-1].lower() if found_directions else None
    
    # Convert found labels to lowercase and ensure they are in the label_to_instance set
    labels = [label.lower() for label in found_labels if label.lower() in label_to_instance]
    
    # Convert list of labels to a comma-separated string in lowercase
    label_string = ", ".join(labels)
    
    return label_string, direction

def keypoint_detection(image, prompt, sel_pix):

    prompt, direction = parse_input_text(prompt)
    if prompt in globals():
        keypoint_dict = globals()[prompt]
        keypoint_text_prompt = keypoint_dict.get("keypoints")
        keypoint_skeleton = keypoint_dict.get("skeleton")

    # Check if the predefined keypoints are loaded correctly
    print(keypoint_text_prompt)  # face 딕셔너리가 제대로 로드되었는지 확인
    parser = argparse.ArgumentParser("UniPose Inference", add_help=True)
    args = parser.parse_args()
    # Construct relative paths for config and checkpoint files
    config_file = os.path.join(script_dir, '..', '..', 'X-Pose', 'config_model', 'UniPose_SwinT.py')
    checkpoint_path = os.path.join(script_dir, '..', '..', 'X-Pose', 'weights', 'unipose_swint.pth')
    # load model
    keypoint_detection_model = load_model_for_gooddrag(config_file, checkpoint_path, args, cpu_only=False)
    
    # load image
    image_pil, image = load_image(image)


    if prompt == "face":
        box_threshhold = 0.05
    elif prompt == "car":
        box_threshhold = 0.1
    else:
        box_threshhold = 0.1
    # run model
    boxes_filt, keypoints_filt = get_unipose_output(
        keypoint_detection_model, image, prompt, keypoint_text_prompt, box_threshhold, 0.9, cpu_only=False 
    ) ## keypoint의 원활한 검출을 위해 box의 임계값을 0.05로 낮춤
    size = image_pil.size
    W, H = size
    tgt = {
        "boxes": boxes_filt,
        "keypoints": keypoints_filt,
        "size": [W, H]
    }
    
    num_kpts = len(keypoint_text_prompt)
    all_keypoints = []  # 키포인트 좌표를 저장할 리스트
    
    for ann in tgt['keypoints']:
        
        kp = np.array(ann.cpu())
        Z = kp[:num_kpts * 2] * np.array([W, H] * num_kpts) # 이미지 크기에 맞게 변환
        x = Z[0::2]
        y = Z[1::2]
        all_keypoints.append((x.tolist(), y.tolist()))  # 키포인트 좌표를 저장
    
    
    for x_array, y_array in all_keypoints:
        for x, y in zip(x_array, y_array):
            x = max(0, min(510, int(x)))
            y = max(0, min(510, int(y)))
            sel_pix.append((x, y))
            if direction:
                offset_x, offset_y = direction_to_offset[direction]
                x_offset = max(0, min(510, int(x) + offset_x))
                y_offset = max(0, min(510, int(y) + offset_y))
                sel_pix.append((x_offset, y_offset))
            else:
                sel_pix.append((x, y))
    
    if isinstance(image_pil, Image.Image):
        image = np.array(image_pil)
                  
    points = []
    for idx, point in enumerate(sel_pix):
        if idx % 2 == 0:
            # draw a red circle at the handle point
            cv2.circle(image, tuple(point), 10, (255, 0, 0), -1)
        else:
            # draw a blue circle at the handle point
            cv2.circle(image, tuple(point), 10, (0, 0, 255), -1)
        points.append(tuple(point))
        # draw an arrow from handle point to target point
        if len(points) == 2:
            cv2.arrowedLine(image, points[0], points[1], (255, 255, 255), 4, tipLength=0.5)
            points = []
    
    return image

## 여기까지 UniPose를 위해 추가한 함수들##########################

# user click the image to get points, and show the points on the image
def get_points(img,
               sel_pix,
               evt: gr.SelectData):
    # collect the selected point
    sel_pix.append(evt.index)
    # draw points
    points = []
    for idx, point in enumerate(sel_pix):
        if idx % 2 == 0:
            # draw a red circle at the handle point
            cv2.circle(img, tuple(point), 5, (255, 0, 0), -1)
        else:
            # draw a blue circle at the handle point
            cv2.circle(img, tuple(point), 5, (0, 0, 255), -1)
        points.append(tuple(point))
        # draw an arrow from handle point to target point
        if len(points) == 2:
            cv2.arrowedLine(img, points[0], points[1], (255, 255, 255), 4, tipLength=0.5)
            points = []
    return img if isinstance(img, np.ndarray) else np.array(img)


def show_cur_points(img,
                    sel_pix,
                    bgr=False):
    # draw points
    points = []
    for idx, point in enumerate(sel_pix):
        if idx % 2 == 0:
            # draw a red circle at the handle point
            red = (255, 0, 0) if not bgr else (0, 0, 255)
            cv2.circle(img, tuple(point), 5, red, -1)
        else:
            # draw a blue circle at the handle point
            blue = (0, 0, 255) if not bgr else (255, 0, 0)
            cv2.circle(img, tuple(point), 5, blue, -1)
        points.append(tuple(point))
        # draw an arrow from handle point to target point
        if len(points) == 2:
            cv2.arrowedLine(img, points[0], points[1], (255, 255, 255), 4, tipLength=0.5)
            points = []
    return img if isinstance(img, np.ndarray) else np.array(img)


# clear all handle/target points
def undo_points(original_image,
                mask):
    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        masked_img = mask_image(original_image, 1 - mask, color=[0, 0, 0], alpha=0.3)
    else:
        masked_img = original_image.copy()
    return masked_img, []


def clear_folder(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Iterate over all the files and directories in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            # Check if it's a file or a directory
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove the file or link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove the directory and all its contents


def train_lora_interface(original_image,
                         prompt,
                         model_path,
                         vae_path,
                         lora_path,
                         lora_step,
                         lora_lr,
                         lora_batch_size,
                         lora_rank,
                         progress=gr.Progress(),
                         use_gradio_progress=True):
    if not os.path.exists(lora_path):
        os.makedirs(lora_path)

    clear_folder(lora_path)

    train_lora(
        original_image,
        prompt,
        model_path,
        vae_path,
        lora_path,
        lora_step,
        lora_lr,
        lora_batch_size,
        lora_rank,
        progress,
        use_gradio_progress)
    return "Training LoRA Done!"


def preprocess_image(image,
                     device):
    image = torch.from_numpy(image).float() / 127.5 - 1  # [-1, 1]
    image = rearrange(image, "h w c -> 1 c h w")
    image = image.to(device)
    return image


def save_images_with_pillow(images, base_filename='image'):
    for index, img in enumerate(images):
        # Convert array to Image object and save
        img_pil = Image.fromarray(img)
        folder_path = f'./save'
        filename = os.path.join(folder_path, "{}_{}.png".format(base_filename, index))
        img_pil.save(filename)
        print(f"Saved: {filename}")


def get_original_points(handle_points: List[torch.Tensor],
                        full_h: int,
                        full_w: int,
                        sup_res_w,
                        sup_res_h,
                        ) -> List[torch.Tensor]:
    """
    Convert local handle points and target points back to their original UI coordinates.

    Args:
        sup_res_h: Half original height of the UI canvas.
        sup_res_w: Half original width of the UI canvas.
        handle_points: List of handle points in local coordinates.
        full_h: Original height of the UI canvas.
        full_w: Original width of the UI canvas.

    Returns:
        original_handle_points: List of handle points in original UI coordinates.
    """
    original_handle_points = []

    for cur_point in handle_points:
        original_point = torch.round(
            torch.tensor([cur_point[1] * full_w / sup_res_w, cur_point[0] * full_h / sup_res_h]))
        original_handle_points.append(original_point)

    return original_handle_points


def save_image_mask_points(mask, points, image_with_points, output_dir='./saved_data'):
    """
    Saves the mask and points to the specified directory.

    Args:
      mask: The mask data as a numpy array.
      points: The list of points collected from the user interaction.
      image_with_points: The image with points clicked by the user.
      output_dir: The directory where to save the data.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save mask
    mask_path = os.path.join(output_dir, f"mask.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)

    # Save points
    points_path = os.path.join(output_dir, f"points.json")
    with open(points_path, 'w') as f:
        json.dump({'points': points}, f)

    image_with_points_path = os.path.join(output_dir, "image_with_points.jpg")
    Image.fromarray(image_with_points).save(image_with_points_path)

    return


def save_drag_result(output_image, new_points, result_path):
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    result_dir = f'{result_path}'
    os.makedirs(result_dir, exist_ok=True)
    output_image_path = os.path.join(result_dir, 'output_image.png')
    cv2.imwrite(output_image_path, output_image)

    img_with_new_points = show_cur_points(np.ascontiguousarray(output_image), new_points, bgr=True)
    new_points_image_path = os.path.join(result_dir, 'image_with_new_points.png')
    cv2.imwrite(new_points_image_path, img_with_new_points)

    points_path = os.path.join(result_dir, f'new_points.json')
    with open(points_path, 'w') as f:
        json.dump({'points': new_points}, f)


def save_intermediate_images(intermediate_images, result_dir):
    for i in range(len(intermediate_images)):
        intermediate_images[i] = cv2.cvtColor(intermediate_images[i], cv2.COLOR_RGB2BGR)
        intermediate_images_path = os.path.join(result_dir, f'output_image_{i}.png')
        cv2.imwrite(intermediate_images_path, intermediate_images[i])


def create_video(image_folder, data_folder, fps=2, first_frame_duration=2, last_frame_extra_duration=2):
    """
    Creates an MP4 video from a sequence of images using OpenCV.
    """
    img_folder = Path(image_folder)
    img_num = len(list(img_folder.glob('*.png')))

    # Path to the original image with points
    data_folder = Path(data_folder)
    original_path = data_folder / 'image_with_points.jpg'
    output_path = img_folder / 'dragging.mp4'
    # Collect all image paths
    img_files = [original_path]

    # Load the first image to determine the size
    frame = cv2.imread(str(img_files[0]))
    height, width, layers = frame.shape
    size = (int(width), int(height))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4 format
    video = cv2.VideoWriter(str(output_path), fourcc, int(fps), size)

    for _ in range(int(fps * first_frame_duration)):
        video.write(frame)

    # Add images to video
    for i in range(img_num - 2):
        video.write(cv2.imread(str(img_folder / f'output_image_{i}.png')))

    last_frame = cv2.imread(str(img_folder / 'output_image.png'))
    for _ in range(int(fps * last_frame_extra_duration)):
        video.write(last_frame)

    video.release()


def run_gooddrag(source_image,
                 image_with_clicks,
                 mask,
                 prompt,
                 points,
                 inversion_strength,
                 lam,
                 latent_lr,
                 model_path,
                 vae_path,
                 lora_path,
                 drag_end_step,
                 track_per_step,
                 r1,
                 r2,
                 d,
                 max_drag_per_track,
                 max_track_no_change,
                 feature_idx=3,
                 result_save_path='',
                 return_intermediate_images=True,
                 drag_loss_threshold=0,
                 save_intermedia=False,
                 compare_mode=False,
                 once_drag=False,
                 ):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    height, width = source_image.shape[:2]
    n_inference_step = 50
    guidance_scale = 1.0
    seed = 42
    dragger = GoodDragger(device, model_path, prompt, height, width, inversion_strength, r1, r2, d,
                          drag_end_step, track_per_step, lam, latent_lr,
                          n_inference_step, guidance_scale, feature_idx, compare_mode, vae_path, lora_path, seed,
                          max_drag_per_track, drag_loss_threshold, once_drag, max_track_no_change)

    source_image = preprocess_image(source_image, device)

    gen_image, intermediate_features, new_points_handle, intermediate_images = \
        dragger.good_drag(source_image, points,
                          mask,
                          return_intermediate_images=return_intermediate_images)

    new_points_handle = get_original_points(new_points_handle, height, width, dragger.sup_res_w, dragger.sup_res_h)
    if save_intermedia:
        drag_image = [dragger.latent2image(i.cuda()) for i in intermediate_features]
        save_images_with_pillow(drag_image, base_filename='drag_image')

    gen_image = F.interpolate(gen_image, (height, width), mode='bilinear')

    out_image = gen_image.cpu().permute(0, 2, 3, 1).numpy()[0]
    out_image = (out_image * 255).astype(np.uint8)

    new_points = []
    for i in range(len(new_points_handle)):
        new_cur_handle_points = new_points_handle[i].numpy().tolist()
        new_cur_handle_points = [int(point) for point in new_cur_handle_points]
        new_points.append(new_cur_handle_points)
        new_points.append(points[i * 2 + 1])

    print(f'points {points}')
    print(f'new points {new_points}')

    if return_intermediate_images:
        os.makedirs(result_save_path, exist_ok=True)
        for i in range(len(intermediate_images)):
            intermediate_images[i] = F.interpolate(intermediate_images[i], (height, width), mode='bilinear')
            intermediate_images[i] = intermediate_images[i].cpu().permute(0, 2, 3, 1).numpy()[0]
            intermediate_images[i] = (intermediate_images[i] * 255).astype(np.uint8)

        for i in range(len(intermediate_images)):
            intermediate_images[i] = cv2.cvtColor(intermediate_images[i], cv2.COLOR_RGB2BGR)
            intermediate_images_path = os.path.join(result_save_path, f'output_image_{i}.png')
            cv2.imwrite(intermediate_images_path, intermediate_images[i])

    return out_image, new_points
