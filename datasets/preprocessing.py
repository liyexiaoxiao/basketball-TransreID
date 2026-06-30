import random
import math
import cv2
import numpy as np
from PIL import Image

# Cache for np.meshgrid results keyed by (H, W) to avoid repeated allocations
_meshgrid_cache = {}


def _get_meshgrid(h, w):
    """Return cached meshgrid for given dimensions."""
    key = (h, w)
    if key not in _meshgrid_cache:
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x, y)
        _meshgrid_cache[key] = (xx, yy)
    return _meshgrid_cache[key]


class ComposeNP:
    """Compose multiple NumPy-based transforms efficiently.

    Converts PIL→NumPy once, runs all transforms in NumPy space,
    converts back to PIL once. This eliminates redundant conversions
    when chaining multiple custom transforms that each operate on
    NumPy arrays internally.

    Args:
        transforms: list of callables, each taking a numpy array and
                    returning a numpy array.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        is_pil = isinstance(img, Image.Image)
        if is_pil:
            img = np.array(img)
        for t in self.transforms:
            img = t(img)
        if is_pil:
            img = Image.fromarray(img)
        return img

    def __repr__(self):
        return 'ComposeNP([' + ', '.join(str(t) for t in self.transforms) + '])'


class RandomMotionBlur(object):
    """ Randomly applies motion blur to an image.
    Uses cv2.line to create an anti-aliased, physically accurate motion kernel.
    Args:
         probability: The probability that the operation will be performed.
         kernel_sizes: The sizes of the motion blur kernel (must be odd).
         angle_range: A tuple defining the range of motion angles (e.g., (0, 360)).
    """
    def __init__(self, probability=0.5, kernel_sizes=[5, 7, 9, 11, 15], angle_range=(0, 360)):
        self.probability = probability
        self.kernel_sizes = kernel_sizes
        self.angle_range = angle_range

    def __call__(self, img):
        if random.uniform(0, 1) >= self.probability:
            return img

        # Work with numpy array (avoid copy when already NumPy)
        if isinstance(img, Image.Image):
            img_np = np.array(img)
            is_pil = True
        else:
            img_np = img
            is_pil = False

        ksize = random.choice(self.kernel_sizes)
        angle = random.uniform(self.angle_range[0], self.angle_range[1])

        # Generate smooth, anti-aliased motion blur kernel
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        center = ksize // 2
        
        theta = np.deg2rad(angle)
        length = ksize / 2.0
        x0 = int(round(center - length * np.cos(theta)))
        y0 = int(round(center - length * np.sin(theta)))
        x1 = int(round(center + length * np.cos(theta)))
        y1 = int(round(center + length * np.sin(theta)))

        cv2.line(kernel, (x0, y0), (x1, y1), 1.0, thickness=1, lineType=cv2.LINE_AA)

        kernel_sum = np.sum(kernel)
        if kernel_sum == 0:
            kernel[center, center] = 1.0
        else:
            kernel /= kernel_sum

        blurred_img_np = cv2.filter2D(img_np, -1, kernel)

        if is_pil:
            return Image.fromarray(blurred_img_np)
        return blurred_img_np


class RandomDirectionalLighting(object):
    """ Simulates directional lighting (e.g. sunlight or spotlight from one side).
    Applies a linear gradient of intensity across the image.
    """
    def __init__(self, probability=0.5, intensity_range=(0.3, 0.6)):
        self.probability = probability
        self.intensity_range = intensity_range

    def __call__(self, img):
        if random.uniform(0, 1) >= self.probability:
            return img
            
        is_pil = isinstance(img, Image.Image)
        img_np = np.array(img) if is_pil else img
        h, w = img_np.shape[:2]
        
        # generate a gradient direction
        angle = random.uniform(0, 2 * np.pi)
        
        # coordinate grid for the image (cached by size)
        key = (h, w, 'linspace')
        if key not in _meshgrid_cache:
            x = np.linspace(-1, 1, w)
            y = np.linspace(-1, 1, h)
            _meshgrid_cache[key] = np.meshgrid(x, y)
        xx, yy = _meshgrid_cache[key]
        
        # gradient: a * x + b * y
        grad = xx * np.cos(angle) + yy * np.sin(angle)  
        grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-6) # normalize to 0-1
        
        # Randomly choose how severe the lighting difference is
        intensity = random.uniform(self.intensity_range[0], self.intensity_range[1])
        
        # Creating a lighting mask that gradients across the image
        mean_brightness = random.uniform(0.8, 1.2)
        light_mask = mean_brightness - intensity + 2 * intensity * grad
        light_mask = np.clip(light_mask, 0.1, 3.0)
        
        # expand mask if color image
        if len(img_np.shape) == 3:
            light_mask = np.expand_dims(light_mask, axis=-1)
            
        # Apply to image
        img_np = img_np.astype(np.float32) * light_mask
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        
        if is_pil:
            return Image.fromarray(img_np)
        return img_np

class RandomColorTemperature(object):
    """ Simulates indoor vs outdoor lighting by randomly shifting color temperature (warm vs cold).
    Indoor lights (halogen/sodium) are warm (orange/yellow).
    Outdoor lights (daylight) are cold (blue/white).
    """
    def __init__(self, probability=0.5, shift_range=30):
        self.probability = probability
        self.shift_range = shift_range

    def __call__(self, img):
        import random, numpy as np
        if random.uniform(0, 1) >= self.probability:
            return img
            
        is_pil = isinstance(img, Image.Image)
        img_np = np.array(img).astype(np.float32) if is_pil else img.astype(np.float32)
        
        if len(img_np.shape) != 3 or img_np.shape[2] != 3:
            return img # Only apply to RGB

        # Shift: positive means warmer (more R, less B), negative means cooler (less R, more B)
        shift = random.uniform(-self.shift_range, self.shift_range)
        
        img_np[:, :, 0] += shift # Red
        img_np[:, :, 2] -= shift # Blue
        
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        
        if is_pil:
            return Image.fromarray(img_np)
        return img_np


class RandomISONoise(object):
    """ Simulates high ISO noise often seen in indoor sports photography due to low light and high shutter speeds.
    """
    def __init__(self, probability=0.5, intensity_range=(10.0, 25.0)):
        self.probability = probability
        self.intensity_range = intensity_range

    def __call__(self, img):
        import random, numpy as np
        if random.uniform(0, 1) >= self.probability:
            return img
            
        is_pil = isinstance(img, Image.Image)
        img_np = np.array(img).astype(np.float32) if is_pil else img.astype(np.float32)
        
        sigma = random.uniform(self.intensity_range[0], self.intensity_range[1])
        noise = np.random.normal(0, sigma, img_np.shape)
        
        img_np = img_np + noise
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        
        if is_pil:
            return Image.fromarray(img_np)
        return img_np


class RandomBackgroundBlur(object):
    """ Simulates shallow depth of field (bokeh) to reduce background interference.
    Maintains a sharp center (where the subject usually is in a ReID bbox)
    and heavily blurs the periphery (where the background typically is).
    """
    def __init__(self, probability=0.5, blur_kernel=(15, 25)):
        self.probability = probability
        self.blur_kernel = blur_kernel

    def __call__(self, img):
        import random, cv2, numpy as np
        if random.uniform(0, 1) >= self.probability:
            return img
            
        is_pil = isinstance(img, Image.Image)
        img_np = np.array(img).astype(np.float32) if is_pil else img.astype(np.float32)
        h, w = img_np.shape[:2]
        
        # Determine a random heavy blur kernel
        k = random.choice(range(self.blur_kernel[0], self.blur_kernel[1]+1, 2))
        blurred_img = cv2.GaussianBlur(img_np, (k, k), 0)
        
        # Create an elliptical mask for the center
        cx = w / 2 + random.uniform(-w*0.1, w*0.1)
        cy = h / 2 + random.uniform(-h*0.1, h*0.1)
        
        rx = w * random.uniform(0.35, 0.5)
        ry = h * random.uniform(0.35, 0.5)
        
        # coordinate grid (cached by size)
        key = (h, w, 'arange')
        if key not in _meshgrid_cache:
            x = np.arange(0, w)
            y = np.arange(0, h)
            _meshgrid_cache[key] = np.meshgrid(x, y)
        xx, yy = _meshgrid_cache[key]
        
        dist = ((xx - cx) / rx)**2 + ((yy - cy) / ry)**2
        
        # Smooth gaussian mask: 1.0 at center, decays to 0.0 towards edges
        mask = np.exp(-dist * 1.5)
        
        if len(img_np.shape) == 3:
            mask = np.expand_dims(mask, axis=-1)
            
        # Blend
        result = img_np * mask + blurred_img * (1.0 - mask)
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        if is_pil:
            return Image.fromarray(result)
        return result

