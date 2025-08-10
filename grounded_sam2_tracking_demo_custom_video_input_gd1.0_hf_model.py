import os
import cv2
import torch
import numpy as np
import supervision as sv

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor 
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images

"""
Hyperparam for Ground and Tracking
"""
MODEL_ID = "IDEA-Research/grounding-dino-tiny"
VIDEO_PATH = "./assets/video.mp4"
TEXT_PROMPT = "person not wearing a white lab coat."
OUTPUT_VIDEO_PATH = "./person_tracking_demo.mp4"
SOURCE_VIDEO_FRAME_DIR = "./custom_video_frames"
SAVE_TRACKING_RESULTS_DIR = "./tracking_results"
PROMPT_TYPE_FOR_VIDEO = "box" # choose from ["point", "box", "mask"]

"""
Step 1: Environment settings and model initialization for SAM 2
"""
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# init sam image predictor and video predictor model
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)

# build grounding dino from huggingface
model_id = MODEL_ID
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)


def is_wearing_lab_coat(frame_bgr: np.ndarray, mask: np.ndarray,
                        debug_mode=False) -> bool:
    """
    Enhanced lab coat detection using multiple color space analysis and texture features.
    Returns True if a significant portion of the person is wearing white/light colored clothing.
    """
    # Ensure mask is binary
    mask_binary = (mask > 0).astype(np.uint8)
    
    # Check if mask has any pixels
    if mask_binary.sum() == 0:
        return False
    
    # Get the person region
    person_region = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask_binary)
    
    # Convert to different color spaces for comprehensive analysis
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    # Extract masked regions
    h_masked = hsv[:, :, 0][mask_binary > 0]
    s_masked = hsv[:, :, 1][mask_binary > 0]
    v_masked = hsv[:, :, 2][mask_binary > 0]
    
    l_masked = lab[:, :, 0][mask_binary > 0]  # Lightness channel
    a_masked = lab[:, :, 1][mask_binary > 0]  # Green-Red channel
    b_masked = lab[:, :, 2][mask_binary > 0]  # Blue-Yellow channel
    
    gray_masked = gray[mask_binary > 0]
    
    total_pixels = len(h_masked)
    if total_pixels == 0:
        return False
    
    # Method 1: HSV-based white detection (multiple thresholds for different lighting)
    # Strict white (bright lighting)
    strict_white = (s_masked < 30) & (v_masked > 200)
    
    # Medium white (normal lighting)
    medium_white = (s_masked < 50) & (v_masked > 160) & (v_masked <= 200)
    
    # Soft white (low lighting or shadows)
    soft_white = (s_masked < 70) & (v_masked > 120) & (v_masked <= 160)
    
    # Very soft white (very low lighting)
    very_soft_white = (s_masked < 90) & (v_masked > 80) & (v_masked <= 120)
    
    hsv_white_pixels = strict_white | medium_white | soft_white | very_soft_white
    hsv_white_ratio = hsv_white_pixels.sum() / total_pixels
    
    # Method 2: LAB-based white detection (better for varying lighting)
    # L* > 50 (bright), a* and b* near 128 (neutral color)
    lab_white_pixels = (l_masked > 120) & (np.abs(a_masked - 128) < 15) & (np.abs(b_masked - 128) < 15)
    lab_bright_pixels = (l_masked > 100) & (np.abs(a_masked - 128) < 25) & (np.abs(b_masked - 128) < 25)
    lab_medium_pixels = (l_masked > 80) & (np.abs(a_masked - 128) < 35) & (np.abs(b_masked - 128) < 35)
    
    lab_white_combined = lab_white_pixels | lab_bright_pixels | lab_medium_pixels
    lab_white_ratio = lab_white_combined.sum() / total_pixels
    
    # Method 3: Grayscale brightness analysis
    bright_gray = gray_masked > 180
    medium_gray = (gray_masked > 140) & (gray_masked <= 180)
    light_gray = (gray_masked > 100) & (gray_masked <= 140)
    
    gray_light_pixels = bright_gray | medium_gray | light_gray
    gray_light_ratio = gray_light_pixels.sum() / total_pixels
    
    # Method 4: BGR color space analysis for whites and off-whites
    b_masked = frame_bgr[:, :, 0][mask_binary > 0]
    g_masked = frame_bgr[:, :, 1][mask_binary > 0]
    r_masked = frame_bgr[:, :, 2][mask_binary > 0]
    
    # Check for balanced RGB values (characteristic of white/gray)
    rgb_mean = (r_masked.astype(np.float32) + g_masked.astype(np.float32) + b_masked.astype(np.float32)) / 3
    rgb_std = np.sqrt(((r_masked - rgb_mean)**2 + (g_masked - rgb_mean)**2 + (b_masked - rgb_mean)**2) / 3)
    
    # White/light pixels have high mean and low standard deviation
    bgr_white_pixels = (rgb_mean > 120) & (rgb_std < 30)
    bgr_light_pixels = (rgb_mean > 90) & (rgb_std < 40)
    bgr_medium_pixels = (rgb_mean > 60) & (rgb_std < 50)
    
    bgr_light_combined = bgr_white_pixels | bgr_light_pixels | bgr_medium_pixels
    bgr_light_ratio = bgr_light_combined.sum() / total_pixels
    
    # Method 5: Focus on upper body region (lab coats are typically on torso)
    height, width = mask_binary.shape
    
    # Create upper body mask (top 70% of the person)
    coords = np.where(mask_binary > 0)
    if len(coords[0]) > 0:
        min_y, max_y = coords[0].min(), coords[0].max()
        person_height = max_y - min_y
        upper_body_cutoff = min_y + int(person_height * 0.7)
        
        upper_body_mask = mask_binary.copy()
        upper_body_mask[upper_body_cutoff:, :] = 0
        
        if upper_body_mask.sum() > 0:
            upper_h = hsv[:, :, 0][upper_body_mask > 0]
            upper_s = hsv[:, :, 1][upper_body_mask > 0]
            upper_v = hsv[:, :, 2][upper_body_mask > 0]
            
            upper_white = (upper_s < 50) & (upper_v > 100)
            upper_white_ratio = upper_white.sum() / len(upper_h) if len(upper_h) > 0 else 0
        else:
            upper_white_ratio = 0
    else:
        upper_white_ratio = 0
    
    # Combine all methods with weighted scoring
    # Weights can be adjusted based on importance
    weights = {
        'hsv': 0.3,
        'lab': 0.25,
        'gray': 0.2,
        'bgr': 0.15,
        'upper_body': 0.1
    }
    
    combined_score = (
        hsv_white_ratio * weights['hsv'] +
        lab_white_ratio * weights['lab'] +
        gray_light_ratio * weights['gray'] +
        bgr_light_ratio * weights['bgr'] +
        upper_white_ratio * weights['upper_body']
    )
    
    # Adaptive threshold based on lighting conditions
    # Estimate overall brightness of the image
    overall_brightness = np.mean(gray[mask_binary > 0])
    
    if overall_brightness > 150:  # Bright lighting
        threshold = 0.15
    elif overall_brightness > 100:  # Normal lighting
        threshold = 0.12
    elif overall_brightness > 50:   # Low lighting
        threshold = 0.08
    else:  # Very low lighting
        threshold = 0.05
    
    is_lab_coat = combined_score >= threshold
    
    if debug_mode:
        print(f"  Debug - HSV white ratio: {hsv_white_ratio:.3f}")
        print(f"  Debug - LAB white ratio: {lab_white_ratio:.3f}")
        print(f"  Debug - Gray light ratio: {gray_light_ratio:.3f}")
        print(f"  Debug - BGR light ratio: {bgr_light_ratio:.3f}")
        print(f"  Debug - Upper body white ratio: {upper_white_ratio:.3f}")
        print(f"  Debug - Combined score: {combined_score:.3f}")
        print(f"  Debug - Overall brightness: {overall_brightness:.1f}")
        print(f"  Debug - Threshold used: {threshold:.3f}")
        print(f"  Debug - Result: {'LAB COAT' if is_lab_coat else 'NO LAB COAT'}")
    
    return is_lab_coat


def visualize_lab_coat_detection(frame_bgr: np.ndarray, mask: np.ndarray, 
                                is_lab_coat: bool, person_id: int):
    """
    Draw bounding box around person with color coding:
    Green = wearing lab coat, Red = not wearing lab coat
    """
    # Find bounding box of the mask
    mask_binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get bounding rectangle for all contours
        all_pts = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(all_pts)
        
        # Choose color: GREEN for lab coat, RED for no lab coat
        color = (0, 255, 0) if is_lab_coat else (0, 0, 255)
        label = f"Person {person_id}: {'Lab Coat' if is_lab_coat else 'No Lab Coat'}"
        
        # Draw thick rectangle
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, thickness=4)
        
        # Add label with background for better visibility
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame_bgr, (x, y - 35), (x + label_size[0] + 10, y), color, -1)
        cv2.putText(frame_bgr, label, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), thickness=2)


"""
Custom video input directly using video files
"""
video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)  # get video info
print(video_info)
frame_generator = sv.get_video_frames_generator(VIDEO_PATH, stride=1, start=0, end=None)

# saving video to frames
source_frames = Path(SOURCE_VIDEO_FRAME_DIR)
source_frames.mkdir(parents=True, exist_ok=True)

with sv.ImageSink(
    target_dir_path=source_frames, 
    overwrite=True, 
    image_name_pattern="{:05d}.jpg"
) as sink:
    for frame in tqdm(frame_generator, desc="Saving Video Frames"):
        sink.save_image(frame)

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(SOURCE_VIDEO_FRAME_DIR)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# init video predictor state
inference_state = video_predictor.init_state(video_path=SOURCE_VIDEO_FRAME_DIR)

ann_frame_idx = 0  # the frame index we interact with

"""
Step 2: Prompt Grounding DINO 1.5 with Cloud API for box coordinates
"""

# prompt grounding dino to get the box coordinates on specific frame
img_path = os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[ann_frame_idx])
image = Image.open(img_path)
inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = grounding_model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.2,
    text_threshold=0.2,
    target_sizes=[image.size[::-1]]
)

boxes = results[0]["boxes"].cpu().numpy()
scores = results[0]["scores"].cpu().numpy()
labels = [results[0]["labels"][i] for i in range(len(scores))]

# Filter for humans only
PERSON_CONF_THRESH = 0.3
person_idxs = [
    i for i, (lbl, sc) in enumerate(zip(labels, scores))
    if lbl == "person" and sc >= PERSON_CONF_THRESH
]

if not person_idxs:
    person_idxs = sorted(
        [i for i, lbl in enumerate(labels) if lbl == "person"],
        key=lambda i: scores[i], reverse=True
    )[:3]  # Take top 3 if no high-confidence detections

input_boxes = boxes[person_idxs]
confidences = [float(scores[i]) for i in person_idxs]
class_names = ["person"] * len(person_idxs)

print(f"Found {len(input_boxes)} person(s) with conf ≥ {PERSON_CONF_THRESH}")

# prompt SAM image predictor to get the mask for the object
image_predictor.set_image(np.array(image.convert("RGB")))

# Get masks for all detected persons
masks, scores_sam, logits = image_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)

# convert the mask shape to (n, H, W)
if masks.ndim == 4:
    masks = masks.squeeze(1)

# Load the current frame for lab coat detection
frame_bgr = cv2.imread(img_path)

"""
Step 2.5: Lab Coat Detection and Classification
"""
print("\n=== Lab Coat Detection Results ===")
lab_coat_results = []
person_labels = []

for i, (box, mask) in enumerate(zip(input_boxes, masks)):
    person_id = i + 1
    print(f"\n--- Analyzing Person {person_id} ---")
    is_lab_coat = is_wearing_lab_coat(frame_bgr, mask, debug_mode=True)
    lab_coat_results.append(is_lab_coat)
    
    print(f"Final Result - Person {person_id}: {'Wearing lab coat' if is_lab_coat else 'NOT wearing lab coat'}")
    
    # Visualize detection on frame (for debugging)
    visualize_lab_coat_detection(frame_bgr.copy(), mask, is_lab_coat, person_id)
    
    # Create labels for all persons (both with and without lab coats)
    if is_lab_coat:
        person_labels.append(f"person_{person_id}_labcoat")
    else:
        person_labels.append(f"person_{person_id}_no_labcoat")

print(f"\n=== Summary ===")
print(f"Total persons detected: {len(input_boxes)}")
lab_coat_count = sum(lab_coat_results)
no_lab_coat_count = len(lab_coat_results) - lab_coat_count
print(f"Wearing lab coats: {lab_coat_count}")
print(f"Not wearing lab coats: {no_lab_coat_count}")

# Keep ALL detected persons for tracking (both with and without lab coats)
OBJECTS = person_labels

# Save detection visualization
debug_frame_path = os.path.join(SAVE_TRACKING_RESULTS_DIR, "debug_detection.jpg")
os.makedirs(SAVE_TRACKING_RESULTS_DIR, exist_ok=True)
cv2.imwrite(debug_frame_path, frame_bgr)

if len(input_boxes) == 0:
    print("No persons found! Creating empty output video.")
    # Create a simple empty video frame
    empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(empty_frame, "NO PERSONS DETECTED", (150, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=2)
    cv2.imwrite(os.path.join(SAVE_TRACKING_RESULTS_DIR, "annotated_frame_00000.jpg"), empty_frame)
    create_video_from_images(SAVE_TRACKING_RESULTS_DIR, OUTPUT_VIDEO_PATH)
    exit()

"""
Step 3: Register each object's positive points to video predictor
"""

assert PROMPT_TYPE_FOR_VIDEO in ["point", "box", "mask"], "SAM 2 video predictor only support point/box/mask prompt"

# If you are using point prompts, we uniformly sample positive points based on the mask
if PROMPT_TYPE_FOR_VIDEO == "point":
    # sample the positive points from mask for each objects
    all_sample_points = sample_points_from_masks(masks=masks, num_points=10)

    for object_id, (label, points) in enumerate(zip(OBJECTS, all_sample_points), start=1):
        labels_points = np.ones((points.shape[0]), dtype=np.int32)
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            points=points,
            labels=labels_points,
        )

# Using box prompt
elif PROMPT_TYPE_FOR_VIDEO == "box":
    for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            box=box,
        )

# Using mask prompt is a more straightforward way
elif PROMPT_TYPE_FOR_VIDEO == "mask":
    for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
        labels_mask = np.ones((1), dtype=np.int32)
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            mask=mask
        )
else:
    raise NotImplementedError("SAM 2 video predictor only support point/box/mask prompts")

"""
Step 4: Propagate the video predictor to get the segmentation results for each frame
"""
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

"""
Step 5: Visualize the segment results across the video and save them
"""

ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}

for frame_idx, segments in video_segments.items():
    img = cv2.imread(os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[frame_idx]))
    
    object_ids = list(segments.keys())
    masks_frame = list(segments.values())
    
    if masks_frame:  # Only process if we have masks
        masks_frame = np.concatenate(masks_frame, axis=0)
        
        # Create custom annotations with lab coat status color coding
        annotated_frame = img.copy()
        
        for i, obj_id in enumerate(object_ids):
            mask_current = masks_frame[i]
            object_label = ID_TO_OBJECTS[obj_id]
            
            # Determine if this person is wearing a lab coat based on the label
            is_wearing_labcoat = "labcoat" in object_label and "no_labcoat" not in object_label
            
            # Re-check lab coat status for current frame (more accurate)
            current_labcoat_status = is_wearing_lab_coat(img, mask_current, debug_mode=False)
            
            # Visualize with color coding: Green for lab coat, Red for no lab coat
            visualize_lab_coat_detection(annotated_frame, mask_current, current_labcoat_status, obj_id)
            
            # Add mask overlay with transparency
            mask_color = np.zeros_like(img)
            if current_labcoat_status:
                mask_color[:, :, 1] = 255  # Green channel for lab coat
                status_text = "LAB COAT"
                text_color = (0, 255, 0)
            else:
                mask_color[:, :, 2] = 255  # Red channel for no lab coat
                status_text = "NO LAB COAT"
                text_color = (0, 0, 255)
            
            # Apply mask overlay with transparency
            mask_binary = (mask_current > 0).astype(np.uint8)
            mask_3d = np.stack([mask_binary, mask_binary, mask_binary], axis=2)
            overlay = cv2.addWeighted(annotated_frame, 0.7, mask_color * mask_3d, 0.3, 0)
            annotated_frame = np.where(mask_3d > 0, overlay, annotated_frame)
            
            # Add status text at top of frame
            cv2.putText(annotated_frame, f"Person {obj_id}: {status_text}", 
                       (10, 30 + (obj_id-1)*40), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, text_color, thickness=2)
        
        # Add frame info
        cv2.putText(annotated_frame, f"Frame: {frame_idx}", (img.shape[1] - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), thickness=2)
        
    else:
        annotated_frame = img.copy()
        cv2.putText(annotated_frame, "NO PERSONS TRACKED", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=2)
    
    cv2.imwrite(os.path.join(SAVE_TRACKING_RESULTS_DIR, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)

"""
Step 6: Convert the annotated frames to video
"""
create_video_from_images(SAVE_TRACKING_RESULTS_DIR, OUTPUT_VIDEO_PATH)

print(f"\nProcessing completed!")
print(f"Debug detection frame saved to: {debug_frame_path}")
print(f"Tracking results saved to: {SAVE_TRACKING_RESULTS_DIR}")
print(f"Output video saved to: {OUTPUT_VIDEO_PATH}")