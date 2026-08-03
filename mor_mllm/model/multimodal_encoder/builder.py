import os
from mor_mllm.constants import HF_CACHE_DIR
from .clip_encoder import CLIPVisionTower
import transformers
a, b, c = transformers.__version__.split('.')[:3]
if a == '4' and int(b) >= 37:
    from .siglip_encoder import SiglipVisionTower
# ============================================================================================================

def build_image_tower(image_tower_cfg, **kwargs):
    image_tower = getattr(image_tower_cfg, 'mm_image_tower', getattr(image_tower_cfg, 'image_tower', None))
    # is_absolute_path_exists = os.path.exists(image_tower)
    if image_tower.startswith("openai") or image_tower.startswith("laion"):
        return CLIPVisionTower(image_tower, args=image_tower_cfg, cache_dir=HF_CACHE_DIR, **kwargs)
    if image_tower.startswith("google"):
        return SiglipVisionTower(image_tower, args=image_tower_cfg, cache_dir=HF_CACHE_DIR, **kwargs)

    raise ValueError(f'Unknown image tower: {image_tower}')

def build_video_tower(video_tower_cfg, **kwargs):
    video_tower = getattr(video_tower_cfg, 'mm_video_tower', getattr(video_tower_cfg, 'video_tower', None))
    raise ValueError(f'Unknown video tower: {video_tower}')
# ============================================================================================================
