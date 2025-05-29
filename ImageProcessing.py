import streamlit as st
import cv2
import numpy as np

# Streamlit page configuration
st.set_page_config(page_title="Image Processing App", layout="wide")

# Sidebar for operation selection and parameters
st.sidebar.title("Image Processing Options")
operation = st.sidebar.selectbox(
    "Select Operation",
    [
        "Original Image",
        "Resize Image",
        "Shift Image",
        "Draw Shape",
        "Gaussian Blur",
        "Thresholding",
        "Canny Edge Detection",
        "Write Text"
    ]
)

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg", "jfif"])

if uploaded_file is not None:
    # Read the uploaded im3ge
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Convert BGR (OpenCV) to RGB (for Streamlit display)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display original image
    st.image(image_rgb, caption="Original Image", width = 300)
    
    # Process based on selected operation
    if operation == "Original Image":
        st.write("Displaying the original uploaded image.")
    
    elif operation == "Resize Image":
        st.sidebar.subheader("Resize Parameters")
        width = st.sidebar.number_input("Width", min_value=1, value=200)
        height = st.sidebar.number_input("Height", min_value=1, value=400)
        if st.sidebar.button("Apply Resize"):
            resized_img = cv2.resize(image, (width, height))
            resized_img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            st.image(resized_img_rgb, caption=f"Resized Image ({width}x{height})", use_column_width=True)
    
    elif operation == "Shift Image":
        st.sidebar.subheader("Shift Parameters")
        x_shift = st.sidebar.number_input("X-axis Shift (pixels)", value=50)
        y_shift = st.sidebar.number_input("Y-axis Shift (pixels)", value=50)
        if st.sidebar.button("Apply Shift"):
            matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
            shifted_img = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
            shifted_img_rgb = cv2.cvtColor(shifted_img, cv2.COLOR_BGR2RGB)
            st.image(shifted_img_rgb, caption=f"Shifted Image (X: {x_shift}, Y: {y_shift})", use_column_width=True)
    
    elif operation == "Draw Shape":
        st.sidebar.subheader("Shape Parameters")
        shape = st.sidebar.selectbox("Select Shape", ["Line", "Circle"])
        
        if shape == "Line":
            start_x = st.sidebar.number_input("Start X", value=560)
            start_y = st.sidebar.number_input("Start Y", value=560)
            end_x = st.sidebar.number_input("End X", value=70)
            end_y = st.sidebar.number_input("End Y", value=70)
            color_r = st.sidebar.slider("Red (Color)", 0, 255, 0)
            color_g = st.sidebar.slider("Green (Color)", 0, 255, 255)
            color_b = st.sidebar.slider("Blue (Color)", 0, 255, 0)
            thickness = st.sidebar.number_input("Thickness", min_value=1, value=1)
            if st.sidebar.button("Draw Line"):
                image_with_shape = cv2.line(
                    image.copy(),
                    (start_x, start_y),
                    (end_x, end_y),
                    (color_b, color_g, color_r),
                    thickness
                )
                image_with_shape_rgb = cv2.cvtColor(image_with_shape, cv2.COLOR_BGR2RGB)
                st.image(image_with_shape_rgb, caption="Image with Line", use_column_width=True)
        
        elif shape == "Circle":
            center_x = st.sidebar.number_input("Center X", value=450)
            center_y = st.sidebar.number_input("Center Y", value=450)
            radius = st.sidebar.number_input("Radius", min_value=1, value=200)
            color_r = st.sidebar.slider("Red (Color)", 0, 255, 0)
            color_g = st.sidebar.slider("Green (Color)", 0, 255, 255)
            color_b = st.sidebar.slider("Blue (Color)", 0, 255, 0)
            thickness = st.sidebar.number_input("Thickness", min_value=1, value=2)
            if st.sidebar.button("Draw Circle"):
                image_with_shape = cv2.circle(
                    image.copy(),
                    (center_x, center_y),
                    radius,
                    (color_b, color_g, color_r),
                    thickness
                )
                image_with_shape_rgb = cv2.cvtColor(image_with_shape, cv2.COLOR_BGR2RGB)
                st.image(image_with_shape_rgb, caption="Image with Circle", use_column_width=True)
    
    elif operation == "Gaussian Blur":
        st.sidebar.subheader("Gaussian Blur Parameters")
        kernel_size = st.sidebar.slider("Kernel Size (odd number)", 1, 99, 21, step=2)
        if st.sidebar.button("Apply Blur"):
            blurred_img = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            blurred_img_rgb = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB)
            st.image(blurred_img_rgb, caption=f"Blurred Image (Kernel: {kernel_size}x{kernel_size})", use_column_width=True)
    
    elif operation == "Thresholding":
        st.sidebar.subheader("Thresholding Parameters")
        threshold_value = st.sidebar.slider("Threshold Value", 0, 255, 117)
        max_val = st.sidebar.slider("Max Value", 0, 255, 255)
        # Convert to grayscale for thresholding
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if st.sidebar.button("Apply Threshold"):
            _, binary_img = cv2.threshold(gray_img, threshold_value, max_val, cv2.THRESH_BINARY)
            # Convert single-channel image to 3 channels for display
            binary_img_rgb = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2RGB)
            st.image(binary_img_rgb, caption=f"Thresholded Image (Value: {threshold_value})", use_column_width=True)
    
    elif operation == "Canny Edge Detection":
        st.sidebar.subheader("Canny Edge Detection Parameters")
        lower_threshold = st.sidebar.slider("Lower Threshold", 0, 255, 100)
        upper_threshold = st.sidebar.slider("Upper Threshold", 0, 255, 255)
        if st.sidebar.button("Apply Edge Detection"):
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_img, lower_threshold, upper_threshold)
            # Convert single-channel edges to 3 channels for display
            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            st.image(edges_rgb, caption="Canny Edge Detection", use_column_width=True)
    
    elif operation == "Write Text":
        st.sidebar.subheader("Text Parameters")
        text = st.sidebar.text_input("Text to Write", "This is dog")
        pos_x = st.sidebar.number_input("Position X", value=0)
        pos_y = st.sidebar.number_input("Position Y", value=100)
        font_scale = st.sidebar.slider("Font Scale", 0.1, 10.0, 5.0)
        color_r = st.sidebar.slider("Red (Color)", 0, 255, 255)
        color_g = st.sidebar.slider("Green (Color)", 0, 255, 255)
        color_b = st.sidebar.slider("Blue (Color)", 0, 255, 255)
        thickness = st.sidebar.number_input("Thickness", min_value=1, value=2)
        if st.sidebar.button("Write Text"):
            image_with_text = cv2.putText(
                image.copy(),
                text,
                (pos_x, pos_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (color_b, color_g, color_r),
                thickness
            )
            image_with_text_rgb = cv2.cvtColor(image_with_text, cv2.COLOR_BGR2RGB)
            st.image(image_with_text_rgb, caption="Image with Text", use_column_width=True)

else:
    st.write("Please upload an image to start processing.")