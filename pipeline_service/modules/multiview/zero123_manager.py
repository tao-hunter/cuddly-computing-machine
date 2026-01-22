from __future__ import annotations

import time
from typing import List

import torch
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

from config import Settings
from logger_config import logger


class Zero123PlusService:
    """
    Zero123++ multi-view generation service.
    Generates 6 consistent views from a single input image in one forward pass.
    
    Output views are at fixed azimuths: 30°, 90°, 150°, 210°, 270°, 330°
    with elevation at 30° (looking slightly down).
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = f"cuda:{settings.trellis_gpu}" if torch.cuda.is_available() else "cpu"
        self.pipeline = None
        self.model_id = settings.zero123_model_id

    async def startup(self) -> None:
        """Load the Zero123++ pipeline."""
        logger.info(f"Loading Zero123++ model: {self.model_id}...")
        
        try:
            self.pipeline = DiffusionPipeline.from_pretrained(
                self.model_id,
                custom_pipeline="sudo-ai/zero123plus-pipeline",
                torch_dtype=torch.float16,
            )
            
            # Use recommended scheduler
            self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.pipeline.scheduler.config,
                timestep_spacing='trailing'
            )
            
            self.pipeline.to(self.device)
            logger.success(f"Zero123++ loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading Zero123++: {e}")
            raise RuntimeError(f"Error loading Zero123++: {e}")

    async def shutdown(self) -> None:
        """Shutdown and free resources."""
        if self.pipeline:
            try:
                self.pipeline.to("cpu")
            except Exception:
                pass
        self.pipeline = None
        logger.info("Zero123PlusService closed.")

    def ensure_ready(self) -> None:
        """Ensure the service is ready."""
        if self.pipeline is None:
            raise RuntimeError("Zero123++ pipeline not initialized.")

    def generate_views(
        self, 
        image: Image.Image,
        num_inference_steps: int = 75,
    ) -> List[Image.Image]:
        """
        Generate 6 multi-view images from a single input image.
        
        Args:
            image: Input image (should be square, min 320x320)
            num_inference_steps: Number of denoising steps (default 75)
            
        Returns:
            List of 6 PIL Images at different viewpoints:
            - View 0: 30° azimuth (front-right)
            - View 1: 90° azimuth (right side)
            - View 2: 150° azimuth (back-right)
            - View 3: 210° azimuth (back-left)
            - View 4: 270° azimuth (left side)
            - View 5: 330° azimuth (front-left)
        """
        self.ensure_ready()
        
        t1 = time.time()
        logger.info("Generating 6 multi-views with Zero123++...")
        
        # Ensure image is square and reasonable size
        image = self._prepare_input(image)
        
        # Run inference
        with torch.inference_mode():
            result = self.pipeline(
                image,
                num_inference_steps=num_inference_steps,
            )
        
        # Result is a single tiled image (3x2 grid of 320x320 views)
        # We need to split it into 6 individual images
        views = self._split_grid(result.images[0])
        
        generation_time = time.time() - t1
        logger.success(f"Zero123++ generated 6 views in {generation_time:.2f}s")
        
        return views

    def _prepare_input(self, image: Image.Image) -> Image.Image:
        """Prepare input image for Zero123++."""
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Make square
        w, h = image.size
        if w != h:
            size = max(w, h)
            new_image = Image.new("RGB", (size, size), (255, 255, 255))
            new_image.paste(image, ((size - w) // 2, (size - h) // 2))
            image = new_image
        
        # Resize to 320x320 (Zero123++ native resolution)
        if image.size[0] != 320:
            image = image.resize((320, 320), Image.Resampling.LANCZOS)
        
        return image

    def _split_grid(self, grid_image: Image.Image) -> List[Image.Image]:
        """
        Split the 3x2 grid output into 6 individual views.
        
        Grid layout (960x640):
        [View0] [View1] [View2]
        [View3] [View4] [View5]
        
        Each view is 320x320.
        """
        views = []
        view_size = 320
        
        # Row 0
        for col in range(3):
            left = col * view_size
            view = grid_image.crop((left, 0, left + view_size, view_size))
            views.append(view)
        
        # Row 1
        for col in range(3):
            left = col * view_size
            view = grid_image.crop((left, view_size, left + view_size, 2 * view_size))
            views.append(view)
        
        return views

    def get_best_views_for_3d(self, views: List[Image.Image]) -> List[Image.Image]:
        """
        Select the best views for 3D reconstruction from the 6 generated views.
        
        Zero123++ outputs at azimuths: 30°, 90°, 150°, 210°, 270°, 330°
        
        For Trellis, we want diverse coverage. Returns:
        - Original front (from input, not Zero123++)
        - View 1 (90° - right side)
        - View 4 (270° - left side)
        - View 2 or 3 (back area)
        """
        return [
            views[4],  # 270° - left side
            views[1],  # 90° - right side
            views[2],  # 150° - back-right (or use views[3] for back-left)
        ]
