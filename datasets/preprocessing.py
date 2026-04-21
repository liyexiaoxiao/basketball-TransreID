import random
import math
from PIL import ImageDraw


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


class BasketballStructuredOcclusion(object):
    """Apply basketball-specific structured occlusion on PIL images."""

    def __init__(self, probability=0.5, fill_colors=None):
        self.probability = probability
        self.fill_colors = fill_colors or [
            (18, 18, 18),
            (40, 40, 40),
            (32, 48, 72),
            (72, 24, 24),
        ]

    def __call__(self, img):
        if random.uniform(0, 1) >= self.probability:
            return img

        width, height = img.size
        if width < 8 or height < 8:
            return img

        aug_img = img.copy()
        draw = ImageDraw.Draw(aug_img)
        color = random.choice(self.fill_colors)
        mode = random.choice([
            self._upper_body_occlusion,
            self._center_number_occlusion,
            self._side_overlap_occlusion,
            self._stripe_occlusion,
        ])
        mode(draw, width, height, color)
        return aug_img

    def _clip_box(self, x1, y1, x2, y2, width, height):
        x1 = max(0, min(int(x1), width - 1))
        y1 = max(0, min(int(y1), height - 1))
        x2 = max(x1 + 1, min(int(x2), width))
        y2 = max(y1 + 1, min(int(y2), height))
        return x1, y1, x2, y2

    def _draw_box(self, draw, width, height, x1, y1, x2, y2, color):
        box = self._clip_box(x1, y1, x2, y2, width, height)
        draw.rectangle(box, fill=color)

    def _upper_body_occlusion(self, draw, width, height, color):
        occ_w = random.uniform(0.45, 0.72) * width
        occ_h = random.uniform(0.12, 0.22) * height
        center_x = random.uniform(0.42, 0.58) * width
        top_y = random.uniform(0.08, 0.28) * height
        self._draw_box(
            draw,
            width,
            height,
            center_x - occ_w / 2,
            top_y,
            center_x + occ_w / 2,
            top_y + occ_h,
            color,
        )

    def _center_number_occlusion(self, draw, width, height, color):
        occ_w = random.uniform(0.20, 0.35) * width
        occ_h = random.uniform(0.12, 0.22) * height
        center_x = random.uniform(0.46, 0.54) * width
        center_y = random.uniform(0.38, 0.58) * height
        self._draw_box(
            draw,
            width,
            height,
            center_x - occ_w / 2,
            center_y - occ_h / 2,
            center_x + occ_w / 2,
            center_y + occ_h / 2,
            color,
        )

    def _side_overlap_occlusion(self, draw, width, height, color):
        occ_w = random.uniform(0.12, 0.22) * width
        occ_h = random.uniform(0.55, 0.88) * height
        from_left = random.random() < 0.5
        top_y = random.uniform(0.08, 0.25) * height
        if from_left:
            x1 = random.uniform(0.0, 0.08) * width
            x2 = x1 + occ_w
        else:
            x2 = random.uniform(0.92, 1.0) * width
            x1 = x2 - occ_w
        self._draw_box(draw, width, height, x1, top_y, x2, top_y + occ_h, color)

    def _stripe_occlusion(self, draw, width, height, color):
        stripe_count = random.randint(1, 3)
        horizontal = random.random() < 0.6
        for _ in range(stripe_count):
            if horizontal:
                stripe_h = random.uniform(0.05, 0.10) * height
                y1 = random.uniform(0.20, 0.82) * height
                self._draw_box(
                    draw,
                    width,
                    height,
                    random.uniform(0.05, 0.18) * width,
                    y1,
                    random.uniform(0.82, 0.95) * width,
                    y1 + stripe_h,
                    color,
                )
            else:
                stripe_w = random.uniform(0.05, 0.10) * width
                x1 = random.uniform(0.16, 0.76) * width
                self._draw_box(
                    draw,
                    width,
                    height,
                    x1,
                    random.uniform(0.08, 0.18) * height,
                    x1 + stripe_w,
                    random.uniform(0.80, 0.94) * height,
                    color,
                )
