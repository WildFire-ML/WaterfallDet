# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 14:49:33 2026

@author: Liam
"""

import numpy as np
from scipy.spatial import ConvexHull, distance_matrix
from tqdm import tqdm

import os
from waterfalldet.utils import load_las, project_las_geospatial


def normalize_point_cloud(pc, ground_points=None, dem=None):
    """
    Normalize point cloud heights by subtracting ground elevation.

    Args:
        pc (np.array): Point cloud [x, y, z]
        ground_points (np.array, optional): Ground classified points [x, y, z]
        dem (np.array, optional): Pre-computed DEM to subtract

    Returns:
        normalized_pc (np.array): Point cloud with normalized heights [x, y, z_normalized]
    """
    if dem is not None:
        # If DEM is provided, use it directly
        # Assumes you have a way to query DEM at each point location
        raise NotImplementedError("DEM-based normalization not implemented in this example")

    if ground_points is not None:
        # Simple approach: subtract minimum ground height in local area
        # For production, use proper DEM interpolation
        normalized_pc = pc.copy()
        # This is simplified - in practice, interpolate ground height at each point
        min_ground = ground_points[:, 2].min()
        normalized_pc[:, 2] = pc[:, 2] - min_ground
        return normalized_pc
    else:
        # Assume already normalized or subtract global minimum
        normalized_pc = pc.copy()
        normalized_pc[:, 2] = pc[:, 2] - pc[:, 2].min()
        return normalized_pc


def segment_trees_li2012(pc, min_height=2.0, search_radius=2.0,
                         adaptive_threshold=True, shape_check=True):
    """
    Segment individual trees using Li et al. (2012) method.

    This method segments trees sequentially from tallest to shortest by:
    1. Finding the global maximum (tree top)
    2. Growing the tree cluster using spacing-based rules
    3. Removing segmented tree and repeating

    Args:
        pc (np.array): Normalized point cloud [x, y, z] where z is height above ground
        min_height (float): Minimum tree height to consider
        search_radius (float): Radius for local maximum search
        adaptive_threshold (bool): Use adaptive spacing threshold based on height
        shape_check (bool): Use shape analysis to reduce over-segmentation

    Returns:
        labels (np.array): Tree ID for each point (0 = not assigned to any tree)
        tree_segments (dict): Dictionary mapping tree_id to point indices
    """

    # Filter points by minimum height
    valid_mask = pc[:, 2] >= min_height
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return np.zeros(len(pc), dtype=int), {}

    # Initialize labels (0 = unassigned)
    labels = np.zeros(len(pc), dtype=int)
    tree_segments = {}

    # Working set of unassigned points
    unassigned = set(valid_indices)

    tree_id = 1

    while len(unassigned) > 0:
        # Find global maximum in remaining points
        unassigned_list = list(unassigned)
        heights = pc[unassigned_list, 2]
        max_idx_local = np.argmax(heights)
        seed_idx = unassigned_list[max_idx_local]

        # Check if remaining highest point is too short
        if pc[seed_idx, 2] < min_height:
            break

        # Grow tree cluster from this seed
        tree_points = grow_tree_cluster(
            pc, seed_idx, unassigned, search_radius,
            adaptive_threshold, shape_check
        )

        # Assign labels
        for idx in tree_points:
            labels[idx] = tree_id
            unassigned.discard(idx)

        tree_segments[tree_id] = list(tree_points)
        tree_id += 1
        print(f'Points remaining: {len(unassigned)}')

    print(f'\n{tree_id - 1} trees segmented!')

    return labels, tree_segments


def grow_tree_cluster(pc, seed_idx, available_points, search_radius=2.0,
                      adaptive_threshold=True, shape_check=True):
    """
    Grow a tree cluster from seed point using Li et al. 2012 rules.

    Args:
        pc (np.array): Point cloud [x, y, z]
        seed_idx (int): Index of seed point (tree top)
        available_points (set): Set of available point indices
        search_radius (float): Search radius for neighbors
        adaptive_threshold (bool): Use height-adaptive spacing threshold
        shape_check (bool): Use convex hull shape analysis

    Returns:
        tree_cluster (set): Set of point indices belonging to this tree
    """

    tree_cluster = {seed_idx}
    nearby_tree_points = set()  # Points classified as other trees

    # Add dummy far point to nearby_tree set
    # (represents other trees that will be found later)

    # Sort all available points by height (top to bottom)
    available_list = [idx for idx in available_points if idx != seed_idx]
    if len(available_list) == 0:
        return tree_cluster

    heights = pc[available_list, 2]
    sorted_indices = [available_list[i] for i in np.argsort(heights)[::-1]]

    # Process points from high to low
    for point_idx in sorted_indices:
        if point_idx in tree_cluster or point_idx in nearby_tree_points:
            continue

        point_height = pc[point_idx, 2]

        # Get spacing threshold based on height
        if adaptive_threshold:
            if point_height >= 15:
                spacing_threshold = 2.0
            else:
                spacing_threshold = 1.5
        else:
            spacing_threshold = 2.0

        # Check if point is local maximum
        is_local_max = check_local_maximum(pc, point_idx, available_points,
                                           search_radius)

        if is_local_max:
            # Could be tree top or branch top
            result = classify_local_maximum(
                pc, point_idx, tree_cluster, nearby_tree_points,
                spacing_threshold, shape_check
            )
            # print(result)
            if result == 'target_tree':
                tree_cluster.add(point_idx)
            elif result == 'other_tree':
                nearby_tree_points.add(point_idx)
            # else: uncertain, skip for now

        else:
            # Non-maximum point - use minimum distance rule
            if len(tree_cluster) > 0 and len(nearby_tree_points) > 0:
                # Calculate minimum distance to target tree
                tree_points_2d = pc[list(tree_cluster), :2]
                point_2d = pc[point_idx, :2].reshape(1, -1)
                dist_to_tree = distance_matrix(point_2d, tree_points_2d).min()

                # Calculate minimum distance to other trees
                nearby_points_2d = pc[list(nearby_tree_points), :2]
                dist_to_others = distance_matrix(point_2d, nearby_points_2d).min()

                # Assign to closest cluster
                if dist_to_tree <= dist_to_others:
                    tree_cluster.add(point_idx)
                else:
                    nearby_tree_points.add(point_idx)

            elif len(tree_cluster) > 0:
                # Only target tree exists, check distance
                tree_points_2d = pc[list(tree_cluster), :2]
                point_2d = pc[point_idx, :2].reshape(1, -1)
                dist_to_tree = distance_matrix(point_2d, tree_points_2d).min()

                if dist_to_tree <= spacing_threshold:
                    tree_cluster.add(point_idx)

    return tree_cluster


def check_local_maximum(pc, point_idx, available_points, search_radius=2.0):
    """
    Check if a point is a local maximum within search radius.

    Args:
        pc (np.array): Point cloud
        point_idx (int): Index of point to check
        available_points (set): Available point indices
        search_radius (float): Search radius

    Returns:
        bool: True if point is local maximum
    """
    point = pc[point_idx]
    point_height = point[2]

    # Find neighbors within search radius
    available_list = list(available_points)
    if len(available_list) == 0:
        return True

    neighbors_2d = pc[available_list, :2]
    point_2d = point[:2].reshape(1, -1)
    distances = distance_matrix(point_2d, neighbors_2d)[0]

    nearby_mask = distances <= search_radius
    nearby_indices = [available_list[i] for i, is_near in enumerate(nearby_mask) if is_near]

    if len(nearby_indices) == 0:
        return True

    # Check if this point is highest in neighborhood
    nearby_heights = pc[nearby_indices, 2]
    return point_height >= nearby_heights.max()


def classify_local_maximum(pc, point_idx, tree_cluster, nearby_tree_points,
                           spacing_threshold, shape_check=True):
    """
    Classify a local maximum point as target tree, other tree, or uncertain.

    Uses spacing and optionally shape analysis per Li et al. 2012.

    Args:
        pc (np.array): Point cloud
        point_idx (int): Point to classify
        tree_cluster (set): Current target tree cluster
        nearby_tree_points (set): Points from other trees
        spacing_threshold (float): Distance threshold
        shape_check (bool): Use convex hull shape analysis

    Returns:
        str: 'target_tree', 'other_tree', or 'uncertain'
    """

    if len(tree_cluster) == 0:
        return 'target_tree'  # First point

    # Calculate minimum distance to target tree (2D)
    tree_points_2d = pc[list(tree_cluster), :2]
    point_2d = pc[point_idx, :2].reshape(1, -1)
    dist_to_tree = distance_matrix(point_2d, tree_points_2d).min()

    # Calculate minimum distance to other trees (2D)
    if len(nearby_tree_points) > 0:
        nearby_points_2d = pc[list(nearby_tree_points), :2]
        dist_to_others = distance_matrix(point_2d, nearby_points_2d).min()
    else:
        dist_to_others = np.inf

    # Rule 1: If far from target tree, likely another tree
    if dist_to_tree > spacing_threshold:
        return 'other_tree'

    # Rule 2: If close to target and closer than to others, likely target tree
    if dist_to_tree <= spacing_threshold and dist_to_tree <= dist_to_others:

        if shape_check and dist_to_tree > spacing_threshold * 0.5:
            # Use shape analysis for borderline cases
            shape_result = analyze_branch_shape(pc, point_idx, tree_cluster)

            if shape_result == 'separate_tree':
                return 'other_tree'
            elif shape_result == 'branch':
                return 'target_tree'

        return 'target_tree'

    # Rule 3: Closer to other trees
    if dist_to_tree > dist_to_others:
        return 'other_tree'

    return 'uncertain'


def analyze_branch_shape(pc, point_idx, tree_cluster, neighbor_radius=1.5):
    """
    Analyze if a point represents a branch or separate tree using convex hull shape.

    Per Li et al. 2012: elongated convex hull suggests branch, compact suggests tree.

    Args:
        pc (np.array): Point cloud
        point_idx (int): Point to analyze
        tree_cluster (set): Current tree cluster
        neighbor_radius (float): Radius to find neighbors of the point

    Returns:
        str: 'branch', 'separate_tree', or 'uncertain'
    """

    # Find neighbors of the candidate point
    point_2d = pc[point_idx, :2]
    tree_list = list(tree_cluster)
    tree_points_2d = pc[tree_list, :2]

    distances = np.sqrt(((tree_points_2d - point_2d) ** 2).sum(axis=1))
    neighbor_mask = distances <= neighbor_radius
    neighbor_indices = [tree_list[i] for i, is_near in enumerate(neighbor_mask) if is_near]

    if len(neighbor_indices) < 3:
        return 'uncertain'

    # Get 2D positions
    neighbor_points_2d = pc[neighbor_indices, :2]

    try:
        # Calculate convex hull
        hull = ConvexHull(neighbor_points_2d)
        perimeter = hull.area  # In 2D, .area gives perimeter
        area = hull.volume      # In 2D, .volume gives area

        if area == 0:
            return 'uncertain'

        # Shape index: SI = P / (4 * sqrt(A))
        # Circle has SI = sqrt(pi) ≈ 1.77
        # More elongated shapes have higher SI
        shape_index = perimeter / (4 * np.sqrt(area))

        # Threshold from paper (approximately)
        if shape_index > 2.5:  # Elongated
            # Check point distribution to determine which tree
            return analyze_point_distribution(pc, point_idx, tree_cluster,
                                              neighbor_indices)
        else:  # Compact
            return 'separate_tree'

    except:
        return 'uncertain'


def analyze_point_distribution(pc, point_idx, tree_cluster, neighbor_indices):
    """
    Analyze distribution of neighbors to determine if branch belongs to target tree.

    Per Li et al. 2012 Figure 7: if most neighbors are on the side toward the target
    tree center, the branch belongs to target tree.

    Args:
        pc (np.array): Point cloud
        point_idx (int): Point to analyze
        tree_cluster (set): Target tree cluster
        neighbor_indices (list): Indices of neighbor points

    Returns:
        str: 'branch' (belongs to target) or 'separate_tree'
    """

    # Calculate gravity center (centroid) of target tree
    tree_points = pc[list(tree_cluster), :2]
    gravity_center = tree_points.mean(axis=0)

    # Vector from point to gravity center
    point_2d = pc[point_idx, :2]
    to_center = gravity_center - point_2d
    center_angle = np.arctan2(to_center[1], to_center[0])

    # Check which side neighbors fall on
    neighbor_points = pc[neighbor_indices, :2]
    to_neighbors = neighbor_points - point_2d
    neighbor_angles = np.arctan2(to_neighbors[:, 1], to_neighbors[:, 0])

    # Calculate angular difference
    angle_diffs = np.abs(neighbor_angles - center_angle)
    angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)

    # Count neighbors on the "center side" (within 90 degrees of center direction)
    center_side_count = np.sum(angle_diffs < np.pi / 2)
    total_count = len(neighbor_indices)

    # If most neighbors are toward center, it's a branch of target tree
    if center_side_count / total_count > 0.6:
        return 'branch'
    else:
        return 'separate_tree'


def save_segmented_trees(pc, labels, out_dir, min_pts=10):
    """
    Save individual tree point clouds to separate files.

    Args:
        pc (np.array): Point cloud [x, y, z]
        labels (np.array): Tree labels for each point
        out_dir (str): Output directory

    Returns:
        saved_files (list): List of saved file paths
    """

    os.makedirs(out_dir, exist_ok=True)
    saved_files = []

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]  # Exclude 0 (unassigned)

    for tree_id in unique_labels:
        mask = labels == tree_id
        tree_points = pc[mask]

        if len(tree_points) < min_pts:
            continue

        out_path = os.path.join(out_dir, f'tree_{int(tree_id)}.xyz')
        np.savetxt(out_path, tree_points, fmt='%.6f')
        saved_files.append(out_path)

    print(f'Saved {len(saved_files)} trees to {out_dir}')
    return saved_files


def tune_search_radius(
    normalized_pc,
    target_trees,
    r_min=0.5,
    r_max=5.0,
    tol=1,
    max_iter=20
):
    """
    Tune the Li 2012 search radius to match a target number of trees.

    Args:
        normalized_pc (np.ndarray): Normalized point cloud.
        target_trees (int): Desired number of tree segments.
        r_min (float): Minimum search radius. Defaults to 0.5.
        r_max (float): Maximum search radius. Defaults to 5.0.
        tol (int): Acceptable absolute error in tree count. Defaults to 1.
        max_iter (int): Maximum number of iterations. Defaults to 20.

    Returns:
        best_radius (float): Search radius that minimizes error.
        best_error (int): Final tree count error.
        best_segments (dict): Tree segments for the best radius.
    """
    best_radius = None
    best_error = np.inf
    best_segments = None

    for _ in range(max_iter):
        r_mid = 0.5 * (r_min + r_max)

        labels, tree_segments = segment_trees_li2012(
            normalized_pc,
            min_height=2.0,
            search_radius=r_mid,
            adaptive_threshold=True,
            shape_check=True
        )

        found_trees = len(tree_segments)
        error = found_trees - target_trees

        if abs(error) < abs(best_error):
            best_radius = r_mid
            best_error = error
            best_segments = tree_segments
            best_labels = labels

        if abs(error) <= tol:
            break

        if error > 0:
            r_min = r_mid
        else:
            r_max = r_mid
    print(f'Best Radius: {best_radius}')
    return best_radius, best_error, best_labels, best_segments
