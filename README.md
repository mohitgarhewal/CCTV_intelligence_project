# CCTV Person Tracking with Grounded SAM 2

## 📌 Project Description
This project uses **Grounded SAM 2** along with **DINO** for **person identification and tracking** in videos. It processes an input video, detects persons based on a prompt, and generates an output video (`person_tracking_demo.mp4`) with tracking overlays. The current model is **pretrained** and demonstrates proof-of-concept functionality for person tracking in lab/CCTV footage.

---

## 🚀 Steps to Run

1. **Run the first block to install dependencies**
   ```bash
   # ▶️ 1. Clone & install

   # 1. Clone the repo
   !git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
   %cd Grounded-SAM-2

   # 2. Install SAM 2 + dependencies
   !pip install git+https://github.com/facebookresearch/segment-anything.git
   !pip install dds-cloudapi-sdk --upgrade
   !pip install supervision opencv-python torch torchvision matplotlib tqdm
   !pip install hydra-core
   !pip install iopath

   # 3. Download pretrained checkpoints
   !bash checkpoints/download_ckpts.sh
   !bash gdino_checkpoints/download_ckpts.sh
   ```

2. **Upload temporary video and run the next block**
   ```bash
   !mv sam2.1_hiera_large.pt checkpoints/sam2.1_hiera_large.pt
   !mv /content/video2.mp4 /content/Grounded-SAM-2/assets/video.mp4
   ```

3. **Change the script name to match your GitHub-uploaded file**  
   Replace:
   ```
   grounded_sam2_tracking_demo_custom_video_input_gd1.0_hf_model.py
   ```
   with your updated script file name from GitHub.  
   Also **verify all parameters** (e.g., video path, prompt, output path).

4. **Run the tracking script**
   ```bash
   VIDEO_PATH="/content/video.mp4"
   TEXT_PROMPT="person"
   OUTPUT_VIDEO_PATH="/content/output.mp4"

   !python grounded_sam2_tracking_demo_custom_video_input_gd1.0_hf_model.py        --VIDEO_PATH="$VIDEO_PATH"        --TEXT_PROMPT="$TEXT_PROMPT"        --OUTPUT_VIDEO_PATH="$OUTPUT_VIDEO_PATH"
   ```

5. **View the output video**  
   The processed video will be saved as:
   ```
   person_tracking_demo.mp4
   ```

---

## 🔮 Further Steps & Improvements

- **Fine-tuning the model**  
  Currently using a pretrained checkpoint. Training with a custom dataset will help increase tracking accuracy.

- **Multi-person detection**  
  The model currently struggles with detecting multiple persons in a single frame. This limitation needs to be addressed for real-world applications.

- **Clothing color extraction improvements**  
  Color extraction can be made more accurate to differentiate between clothing types and enhance detection reliability.
