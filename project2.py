import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import io
from diffusers import StableDiffusionPipeline
import torch
import uuid

# Module 1: Image Generation from Prompt
@st.cache_resource
def load_diffusion_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cpu")  # Use CPU for Pyodide compatibility
    return pipe

def generate_image_from_prompt(prompt):
    pipe = load_diffusion_model()
    image = pipe(prompt).images[0]
    return image

# Module 2: Image Processing
def resize_image(image, width, height):
    return image.resize((width, height), Image.Resampling.LANCZOS)

def crop_image(image, x, y, width, height):
    return image.crop((x, y, x + width, y + height))

def apply_edge_detection(image):
    image_np = np.array(image)
    if len(image_np.shape) == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(image_np, 100, 200)
    return Image.fromarray(edges)

def replace_background(image, bg_color, new_bg_color):
    # Simple chroma-keying: replace bg_color with new_bg_color
    image_np = np.array(image)
    if len(image_np.shape) != 3:
        return image  # Skip if not RGB
    bg_color = np.array(bg_color)
    new_bg_color = np.array(new_bg_color)
    mask = np.all(np.abs(image_np - bg_color) < 50, axis=2)
    image_np[mask] = new_bg_color
    return Image.fromarray(image_np)

# Module 3: Video Generation
def create_video_from_image(image, output_path="output.mp4", duration=5, fps=30):
    image_np = np.array(image)
    height, width = image_np.shape[:2]
    
    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create a simple panning effect
    total_frames = duration * fps
    for i in range(total_frames):
        # Simulate panning by cropping a moving window
        crop_width = int(width * 0.8)
        crop_height = int(height * 0.8)
        x_offset = int((width - crop_width) * (i / total_frames))
        y_offset = int((height - crop_height) * (i / total_frames))
        frame = image_np[y_offset:y_offset + crop_height, x_offset:x_offset + crop_width]
        frame = cv2.resize(frame, (width, height))
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    out.release()
    return output_path

# Module 4: Main Application
def main():
    st.title("Image-to-Video Editor")
    st.write("Generate an image from a text prompt, edit it, and create a video.")
    
    # Image Generation
    st.header("Step 1: Generate Image")
    prompt = st.text_input("Enter a prompt for image generation", "A serene mountain landscape")
    if st.button("Generate Image"):
        with st.spinner("Generating image..."):
            image = generate_image_from_prompt(prompt)
            st.session_state["image"] = image
            st.image(image, caption="Generated Image", use_column_width=True)
    
    # Image Editing
    if "image" in st.session_state:
        image = st.session_state["image"]
        st.header("Step 2: Edit Image")
        
        # Image Processing Options
        st.sidebar.header("Image Editing Options")
        resize = st.sidebar.checkbox("Resize Image")
        if resize:
            width = st.sidebar.slider("Width", 100, 1000, 512)
            height = st.sidebar.slider("Height", 100, 1000, 512)
            image = resize_image(image, width, height)
        
        crop = st.sidebar.checkbox("Crop Image")
        if crop:
            x = st.sidebar.slider("X", 0, image.width - 100, 0)
            y = st.sidebar.slider("Y", 0, image.height - 100, 0)
            crop_width = st.sidebar.slider("Crop Width", 50, image.width - x, 100)
            crop_height = st.sidebar.slider("Crop Height", 50, image.height - y, 100)
            image = crop_image(image, x, y, crop_width, crop_height)
        
        edge_detection = st.sidebar.checkbox("Apply Edge Detection")
        if edge_detection:
            image = apply_edge_detection(image)
        
        bg_replace = st.sidebar.checkbox("Replace Background")
        if bg_replace:
            bg_color = st.sidebar.color_picker("Background Color to Replace", "#00FF00")
            new_bg_color = st.sidebar.color_picker("New Background Color", "#0000FF")
            bg_color_rgb = tuple(int(bg_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
            new_bg_color_rgb = tuple(int(new_bg_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
            image = replace_background(image, bg_color_rgb, new_bg_color_rgb)
        
        st.image(image, caption="Edited Image", use_column_width=True)
        st.session_state["edited_image"] = image
        
        # Download Edited Image
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        st.download_button(
            label="Download Edited Image",
            data=buffered.getvalue(),
            file_name="edited_image.png",
            mime="image/png"
        )
    
    # Video Generation
    if "edited_image" in st.session_state:
        st.header("Step 3: Generate Video")
        duration = st.slider("Video Duration (seconds)", 1, 10, 5)
        if st.button("Generate Video"):
            with st.spinner("Generating video..."):
                video_path = f"output_{uuid.uuid4().hex}.mp4"
                create_video_from_image(st.session_state["edited_image"], video_path, duration)
                with open(video_path, "rb") as f:
                    video_bytes = f.read()
                st.video(video_bytes)
                st.download_button(
                    label="Download Video",
                    data=video_bytes,
                    file_name="output_video.mp4",
                    mime="video/mp4"
                )

if __name__ == "__main__":
    main()