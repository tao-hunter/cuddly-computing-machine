from __future__ import annotations

import time
import numpy as np
from PIL import Image
from rembg import remove, new_session

from config import Settings
from logger_config import logger


class BackgroundRemovalService:
    def __init__(self, settings: Settings):
        """
        Initialize the BackgroundRemovalService using rembg.
        """
        self.settings = settings

        # Set padding percentage, output size
        self.padding_percentage = self.settings.padding_percentage
        self.output_size = self.settings.output_image_size
        self.limit_padding = self.settings.limit_padding

        # rembg session
        self.session = None
        self.model_name = self.settings.rembg_model_name
       
    async def startup(self) -> None:
        """
        Startup the BackgroundRemovalService.
        """
        logger.info(f"Loading rembg model: {self.model_name}...")

        try:
            # Initialize rembg session with specified model
            self.session = new_session(self.model_name)
            logger.success(f"rembg model '{self.model_name}' loaded.")
        except Exception as e:
            logger.error(f"Error loading rembg model: {e}")
            raise RuntimeError(f"Error loading rembg model: {e}")

    async def shutdown(self) -> None:
        """
        Shutdown the BackgroundRemovalService.
        """
        self.session = None
        logger.info("BackgroundRemovalService closed.")

    def ensure_ready(self) -> None:
        """
        Ensure the BackgroundRemovalService is ready.
        """
        if self.session is None:
            raise RuntimeError("rembg session not initialized.")

    def remove_background(self, image: Image.Image) -> Image.Image:
        """
        Remove the background from the image using rembg.
        """
        try:
            t1 = time.time()
            
            # Check if the image already has alpha channel with transparency
            has_alpha = False
            if image.mode == "RGBA":
                alpha = np.array(image)[:, :, 3]
                if not np.all(alpha == 255):
                    has_alpha = True
            
            if has_alpha:
                # If the image has alpha channel, use it directly
                rgba_image = image
            else:
                # Remove background using rembg
                rgba_image = remove(image, session=self.session)
            
            # Crop and resize to output size
            output_image = self._crop_and_resize(rgba_image)
            
            removal_time = time.time() - t1
            logger.success(f"Background remove - Time: {removal_time:.2f}s - OutputSize: {output_image.size} - InputSize: {image.size}")

            return output_image
            
        except Exception as e:
            logger.error(f"Error removing background: {e}")
            return image

    def _crop_and_resize(self, rgba_image: Image.Image) -> Image.Image:
        """
        Crop to bounding box with padding and resize to output size.
        """
        # Get alpha channel for bounding box calculation
        alpha = np.array(rgba_image)[:, :, 3]
        
        # Find bounding box of non-transparent pixels
        rows = np.any(alpha > 128, axis=1)
        cols = np.any(alpha > 128, axis=0)
        
        if not rows.any() or not cols.any():
            # No foreground detected, return resized image
            return rgba_image.resize(self.output_size, Image.Resampling.LANCZOS)
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        # Calculate center and size with padding
        width = x_max - x_min
        height = y_max - y_min
        center_x = (x_max + x_min) / 2
        center_y = (y_max + y_min) / 2
        size = int(max(width, height) * (1 + self.padding_percentage))
        
        # Calculate crop box
        left = int(center_x - size // 2)
        top = int(center_y - size // 2)
        right = int(center_x + size // 2)
        bottom = int(center_y + size // 2)
        
        if self.limit_padding:
            left = max(0, left)
            top = max(0, top)
            right = min(rgba_image.width, right)
            bottom = min(rgba_image.height, bottom)
        
        # Crop and resize
        cropped = rgba_image.crop((left, top, right, bottom))
        return cropped.resize(self.output_size, Image.Resampling.LANCZOS)
