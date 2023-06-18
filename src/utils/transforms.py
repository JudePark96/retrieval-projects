import PIL
import PIL.Image
import torchvision.transforms.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# PIL.Image.MAX_IMAGE_PIXELS = None


def _convert_image_to_rgb(image: PIL.Image):
  return image.convert("RGB")


class SquarePad:
  def __init__(self, size: int):
    self.size = size

  def __call__(self, image):
    w, h = image.size
    max_wh = max(w, h)
    hp = int((max_wh - w) / 2)
    vp = int((max_wh - h) / 2)
    padding = [hp, vp, hp, vp]
    return F.pad(image, padding, 0, 'constant')


class TargetPad:
  def __init__(self, target_ratio: float, size: int):
    self.size = size
    self.target_ratio = target_ratio

  def __call__(self, image):
    w, h = image.size
    actual_ratio = max(w, h) / min(w, h)
    if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
      return image
    scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
    hp = max(int((scaled_max_wh - w) / 2), 0)
    vp = max(int((scaled_max_wh - h) / 2), 0)
    padding = [hp, vp, hp, vp]
    return F.pad(image, padding, 0, 'constant')


def squarepad_transform(dim: int):
  return Compose([
    SquarePad(dim),
    Resize(dim, interpolation=PIL.Image.BICUBIC),
    CenterCrop(dim),
    _convert_image_to_rgb,
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
  ])


def targetpad_transform(target_ratio: float, dim: int):
  return Compose([
    TargetPad(target_ratio, dim),
    Resize(dim, interpolation=PIL.Image.BICUBIC),
    CenterCrop(dim),
    _convert_image_to_rgb,
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
  ])
