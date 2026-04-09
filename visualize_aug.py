import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

# Fix import path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch # Need torch for RandomErasing
from timm.data.random_erasing import RandomErasing

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
    
    # Initialize our transforms with Probability = 1.0 (or as high as possible)
    lighting_aug = RandomDirectionalLighting(probability=1.0, intensity_range=(0.4, 0.8))
    motion_blur_aug = RandomMotionBlur(probability=1.0, kernel_sizes=[11, 15], angle_range=(0, 360))
    color_temp_aug = RandomColorTemperature(probability=1.0, shift_range=40)
    iso_noise_aug = RandomISONoise(probability=1.0, intensity_range=(20.0, 35.0))
    bg_blur_aug = RandomBackgroundBlur(probability=1.0, blur_kernel=(15, 25))
    
    # Standard transforms
    flip_aug = T.RandomHorizontalFlip(p=1.0)
    pad_crop_aug = T.Compose([T.Pad(10), T.RandomCrop(orig_img.size[::-1])])
    color_jitter_aug = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.0)
    gaussian_blur_aug = T.GaussianBlur(kernel_size=5, sigma=(1.0, 2.0))
    random_erasing_aug = RandomErasing(probability=1.0, mode='pixel', max_count=1, device='cpu')
    
    print(f"Processing image: {img_path}")
    
    # Apply augmentations
    img_lighting = lighting_aug(orig_img)
    img_motion_blur = motion_blur_aug(orig_img)
    img_color_temp = color_temp_aug(orig_img)
    img_iso_noise = iso_noise_aug(orig_img)
    img_bg_blur = bg_blur_aug(orig_img)
    
    img_flip = flip_aug(orig_img)
    img_pad_crop = pad_crop_aug(orig_img)
    img_color_jitter = color_jitter_aug(orig_img)
    img_gaussian_blur = gaussian_blur_aug(orig_img)

    # Random Erasing needs tensor
    tensor_img = T.ToTensor()(orig_img)
    tensor_erased = random_erasing_aug(tensor_img.unsqueeze(0)).squeeze(0)
    img_erasing = T.ToPILImage()(tensor_erased)

    # Combined all
    img_combined = color_temp_aug(orig_img)
    img_combined = lighting_aug(img_combined)
    img_combined = bg_blur_aug(img_combined)
    img_combined = iso_noise_aug(img_combined)
    img_combined = motion_blur_aug(img_combined)

    # Plot everything in 3x4 grid
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    images = [
        ("Original", orig_img),
        ("Horiz Flip", img_flip),
        ("Pad + Crop", img_pad_crop),
        ("Random Erasing", img_erasing),
        ("Color Jitter (TV)", img_color_jitter),
        ("Gaussian Blur (TV)", img_gaussian_blur),
        ("Directional Light", img_lighting),
        ("Color Temp (Indoor)", img_color_temp),
        ("ISO Noise", img_iso_noise),
        ("Background Blur", img_bg_blur),
        ("Motion Blur", img_motion_blur),
        ("Combined Everything", img_combined),
    ]
    
    for ax, (title, img) in zip(axes, images):
        ax.imshow(img)
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        
    plt.tight_layout()
    output_path = "augmentation_demo_all.jpg"
    plt.savefig(output_path, dpi=150)
    print(f"Visualization perfectly saved at: {os.path.abspath(output_path)}")

if __name__ == '__main__':
    visualize_augmentations()
