"""
density.py
TraffiSense Vision Module

Adaptive Road Density Estimation
Professional Continuous Heatmap
"""

import cv2
import numpy as np


class DensityMap:

    def __init__(self):

        self.kernel_cache = {}

    def _gaussian_kernel(self, radius):

        if radius in self.kernel_cache:
            return self.kernel_cache[radius]

        size = radius * 2 + 1

        kernel = cv2.getGaussianKernel(size, radius / 2)

        kernel = kernel @ kernel.T

        kernel = kernel / kernel.max()

        self.kernel_cache[radius] = kernel

        return kernel

    def compute(
        self,
        vehicles,
        frame_shape,
        roi_mask=None,
    ):

        h, w = frame_shape[:2]

        density_map = np.zeros((h, w), dtype=np.float32)

        # ------------------------------
        # Build heat map over FULL ROAD
        # ------------------------------

        for vehicle in vehicles:

            x, y = vehicle.center

            x = int(np.clip(x, 0, w - 1))
            y = int(np.clip(y, 0, h - 1))

            name = vehicle.class_name.lower()

            if name == "truck":
                radius = 55
                weight = 3.0

            elif name == "bus":
                radius = 50
                weight = 2.6

            elif name == "car":
                radius = 42
                weight = 1.6

            elif name == "motorcycle":
                radius = 30
                weight = 0.9

            elif name == "bicycle":
                radius = 24
                weight = 0.7

            else:
                radius = 20
                weight = 0.6

            kernel = self._gaussian_kernel(radius)

            k = radius

            x1 = max(0, x - k)
            y1 = max(0, y - k)

            x2 = min(w, x + k + 1)
            y2 = min(h, y + k + 1)

            kx1 = k - (x - x1)
            ky1 = k - (y - y1)

            kx2 = k + (x2 - x)
            ky2 = k + (y2 - y)

            density_map[
                y1:y2,
                x1:x2,
            ] += kernel[
                ky1:ky2,
                kx1:kx2,
            ] * weight

        density_map = cv2.GaussianBlur(
            density_map,
            (0, 0),
            sigmaX=18,
            sigmaY=18,
        )

        # -----------------------------
        # ROI only for calculation
        # -----------------------------

        if roi_mask is None:

            raw_density = float(np.sum(density_map))

        else:

            raw_density = float(
                np.sum(
                    density_map * (roi_mask > 0)
                )
            )

        normalized_density = min(
            raw_density / 4500,
            1.0,
        )

        return {

            "density_map": density_map,

            "raw_density": raw_density,

            "normalized_density": normalized_density,

        }

    def visualize(
        self,
        frame,
        density_map,
        alpha=0.55,
    ):

        if density_map.max() <= 0:
            return frame

        heat = cv2.normalize(

            density_map,

            None,

            0,

            255,

            cv2.NORM_MINMAX,

        ).astype(np.uint8)

        heat = cv2.GaussianBlur(
            heat,
            (0, 0),
            5,
        )

        heat = cv2.applyColorMap(
            heat,
            cv2.COLORMAP_TURBO,
        )

        output = frame.copy()

        mask = cv2.cvtColor(
            heat,
            cv2.COLOR_BGR2GRAY,
        ) > 15

        output[mask] = cv2.addWeighted(

            frame[mask],

            1 - alpha,

            heat[mask],

            alpha,

            0,

        )

        return output