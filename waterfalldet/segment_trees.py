# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 13:34:49 2026

@author: Liam
"""

import os
import numpy as np


def segment_trees(pc, L, resolution, pc_bounds, out_dir):
    """
    Segments a point cloud into individual trees using a 2D segmentation label array
    and saves each tree as a separate .xyz file.

    Args:
        pc (np.ndarray): Point cloud array of shape (N, 3) with columns [x, y, z].
        L (np.ndarray): 2D segmentation array where each tree has a unique integer label (>0).
        resolution (float): Pixel resolution used to rasterize the point cloud.
        pc_bounds (list): [xmin, ymin, xmax, ymax] bounds used to construct L.
        out_dir (str): Path to save segmented trees

    Returns:
        saved_files (list): List of file paths to the saved .xyz tree point clouds.
    """
    os.makedirs(out_dir, exist_ok=True)

    pc = np.asarray(pc)
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]

    # Unpack point cloud bounds
    xmin, ymin, xmax, ymax = pc_bounds

    # Compute pixel indices from world coordinates
    px = np.floor((x - xmin) * resolution).astype(int)
    py = np.floor((y - ymin) * resolution).astype(int)

    # Get raster shape from segmentation array
    w = L.shape[0]
    h = L.shape[1]

    # Create a mask for points that fall within raster bounds
    valid_mask = (px >= 0) & (px < w) & (py >= 0) & (py < h)

    # Filter pixel x indices to valid range
    px = px[valid_mask]
    py = py[valid_mask]

    # Filter point cloud x coordinates to valid range
    x = x[valid_mask]
    y = y[valid_mask]
    z = z[valid_mask]

    # Index segmentation labels at each point location
    labels = L[px, py]

    # Get unique nonzero tree labels
    unique_labels = np.unique(labels[labels > 0])

    # Initialize list to store output file paths
    saved_files = []

    for lab in unique_labels:
        # Create a mask for points belonging to the current tree
        tree_mask = labels == lab

        # Stack the x, y, z coordinates for the current tree
        tree_points = np.column_stack((x[tree_mask], y[tree_mask], z[tree_mask]))

        # Skip if no points were found for this label
        if tree_points.size == 0:
            continue

        # Save
        out_path = os.path.join(out_dir, f'tree_{int(lab)}.xyz')
        np.savetxt(out_path, tree_points, fmt='%.6f')
        saved_files.append(out_path)

    return saved_files
