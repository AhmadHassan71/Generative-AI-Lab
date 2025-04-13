import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import time
import random

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, output_nc, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        model = [
            nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(256, 512, 4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x

def add_imperfections(image, generator):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = generator(img_tensor)

    # Denormalize and convert to PIL Image
    output = (output * 0.5 + 0.5).clamp(0, 1)
    output_image = transforms.ToPILImage()(output.squeeze(0))

    # Apply random transformations
    if random.random() > 0.5:
        output_image = transforms.functional.adjust_brightness(output_image, random.uniform(0.8, 1.2))
    if random.random() > 0.5:
        output_image = transforms.functional.adjust_contrast(output_image, random.uniform(0.8, 1.2))

    return output_image.resize(image.size)

def generate_sketch(image, generator):
    time.sleep(2)  # Simulating processing time
    sketch = add_imperfections(image, generator)
    return sketch

def generate_image_from_sketch(sketch, generator):
    time.sleep(4)  # Simulating processing time
    generated_image = add_imperfections(sketch, generator)
    return generated_image

# Initialize generators
G_photo2sketch = Generator(input_nc=3, output_nc=3)
G_sketch2photo = Generator(input_nc=3, output_nc=3)

# In a real scenario, you would load pre-trained weights here
# G_photo2sketch.load_state_dict(torch.load('photo2sketch_generator.pth'))
# G_sketch2photo.load_state_dict(torch.load('sketch2photo_generator.pth'))

G_photo2sketch.eval()
G_sketch2photo.eval()

st.title("CycleGAN Image Generator App")

# First image input
uploaded_image = st.file_uploader("Upload an image to generate a sketch", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Generate Sketch"):
        sketch = generate_sketch(image, G_photo2sketch)
        st.image(sketch, caption="Generated Sketch", use_column_width=True)

