import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

# Fix import path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.preprocessing import RandomMotionBlur, RandomDirectionalLighting, RandomColorTemperature, RandomISONoise, RandomBackgroundBlur

def visualize_augmentations():
    data_dir = 'data/BallShow/bounding_box_train'
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found!")
        return
        
    img_names = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not img_names:
        print("No images found for testing.")
        return
        
    # Get a random image
    img_path = os.path.join(data_dir, random.choice(img_names))
    orig_img = Image.open(img_path).convert('RGB')
    
    # Initialize our transforms with Probability = 1.0
    lighting_aug = RandomDirectionalLighting(probability=1.0, intensity_range=(0.4, 0.8))
    motion_blur_aug = RandomMotionBlur(probability=1.0, kernel_sizes=[9, 11, 15], angle_range=(0, 360))
    color_temp_aug = RandomColorTemperature(probability=1.0, shift_range=40)
    iso_noise_aug = RandomISONoise(probability=1.0, intensity_range=(20.0, 35.0))
    bg_blur_aug = RandomBackgroundBlur(probability=1.0, blur_kernel=(15, 25))
    
    print(f"Processing image: {img_path}")
    
    # Apply augmentations (generate different versions)
    img_lighting = lighting_aug(orig_img)
    img_motion_blur = motion_blur_aug(orig_img)
    img_color_temp = color_temp_aug(orig_img)
    img_iso_noise = iso_noise_aug(orig_img)
    img_bg_blur = bg_blur_aug(orig_img)

    # Combined Enhanced
    img_combined = motion_blur_aug(iso_noise_aug(color_temp_aug(orig_img)))

    # Plot everything
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    images = [
        ("Original", orig_img),
        ("Directional Light", img_lighting),
        ("Motion Blur", img_motion_blur),
        ("Color Temp", img_color_temp),
        ("ISO Noise", img_iso_noise),
        ("Background Blur (Bokeh)", img_bg_blur),
        ("Combined M+N+C", img_combined),
        ("Combined + Bokeh", bg_blur_aug(img_combined))
    ]
    
    for ax, (title, img) in zip(axes, images):
        ax.imshow(img)
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        
    plt.tight_layout()
    output_path = "augmentation_demo.jpg"
    plt.savefig(output_path, dpi=150)
    print(f"Visualization perfectly saved at: {os.path.abspath(output_path)}")

if __name__ == '__main__':
    visualize_augmentations()
