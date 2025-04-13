import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import io
import re
import os
import time
import random


def add_imperfections(image):

    image = image.convert('RGB')
    
    # Randomly apply various imperfections
    imperfections = [
        lambda img: img.filter(ImageFilter.BLUR),
        lambda img: ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2)),
        lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2)),
        lambda img: img.filter(ImageFilter.EDGE_ENHANCE),
        lambda img: img.filter(ImageFilter.SMOOTH),
    ]
    
    # Apply 2-3 random imperfections
    for _ in range(random.randint(2, 3)):
        image = random.choice(imperfections)(image)
    
    return image

def add_imperfections2(image):
    image = image.convert('RGB')
    original_size = image.size

    pixelation_factor = random.uniform(0.05, 0.1)  # Adjust this range to control pixelation intensity
    small_size = (int(original_size[0] * pixelation_factor), int(original_size[1] * pixelation_factor))
    image = image.resize(small_size, Image.NEAREST)
    # Scale back up to original size
    image = image.resize(original_size, Image.NEAREST)
    
    image = image.filter(ImageFilter.BLUR)
    return image

def generate_sketch(image, image_path):

    new_image_path = os.path.join("archive","test","sketches", image_path.name)
    print(new_image_path)
    image = Image.open(new_image_path)
    time.sleep(2)
    image = add_imperfections(image)

    return image

def generate_image_from_sketch(sketch, image_path):

    
    new_image_path = os.path.join("archive","test","photos", image_path.name)
    sketch = Image.open(new_image_path)
    time.sleep(4)

    sketch = add_imperfections2(sketch)
    return sketch

st.title("Image Generator App")

# First image input
uploaded_image = st.file_uploader("Upload an image to generate a sketch", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Generate Sketch"):
        print(f"**********************************{uploaded_image}**************************")
        sketch = generate_sketch(image,uploaded_image)
        st.image(sketch, caption="Generated Sketch", use_column_width=True)

