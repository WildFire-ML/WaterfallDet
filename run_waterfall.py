# -*- coding: utf-8 -*-
"""
Created on Tue May  2 12:06:58 2023

@author: Liam
"""

# Imports
import argparse
from waterfalldet.waterfall_det import create_dtm, create_chm, create_cmm, gauss_filter, detect_local_maxima, seed_pts_on_image, area_growing, grown_trees_on_image
from waterfalldet.utils import load_las, project_las_geospatial
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--las-file', type=str, help = 'path to the las file')
    parser.add_argument('--resolution', type=int, default = 1, help = 'resolution to create the dtm and chm')
    parser.add_argument('--window-size', default = 1, type=int, help = 'window size to use for local maxima search')
    parser.add_argument('--min_height', default = 2, type=int, help = 'minimum height to call a detection a tree')
    parser.add_argument('--smooth-factor', default = 3, type=int, help = 'kernel to use for gaussian blur on cmm for smoothing')
    parser.add_argument('--th', default = 28, type=float, help = 'initial threshold for area growing (from paper)')
    parser.add_argument('--thStep', default = 0.5, type=float, help = 'threshold step for area growing (from paper)')
    parser.add_argument('--thmin', default = 1, type=float, help = 'minimum threshold for area growing (from paper)')
    
    args = parser.parse_args()
    
    # 1. Import LiDAR
    print('Importing LiDAR...\n')
    h, t = os.path.split(args.las_file)
    pc, l_t_d = load_las(args.las_file)
    pc = project_las_geospatial(pc, l_t_d)
    print('LiDAR imported!\n')
    
    # 2. Create DTM
    print('Creating DTM...\n')
    dtm = create_dtm(pc, args.resolution, save_path = os.path.join(h,'dtm.tif'))
    print('DTM created!\n')
    
    # 3. Create CHM
    print('Creating CHM...\n')
    chm = create_chm(pc, args.resolution, save_path = os.path.join(h,'chm.tif'))
    print('CHM created!\n')
    
    # 4. Create CMM
    print('Creating CMM...\n')
    cmm = create_cmm(chm, dtm)
    print('CMM created!\n')
    
    # 5. CMM Smoothing
    print('Smoothing CMM...\n')
    smoothed_cmm = gauss_filter(cmm, args.smooth_factor)
    print('CMM smoothed!\n')
    
    # 6. Detect Seeds
    print('Detecting seed points...\n')
    seeds = detect_local_maxima(smoothed_cmm, window_size = args.window_size, min_height = args.min_height)
    print(f'{len(seeds)} seeds detected!\n')
    
    # 7. Draw Seeds
    print('Drawing seed points...\n')
    seed_pts_on_image(seeds, smoothed_cmm, cmm, save_path = os.path.join(h, 'seeds.png'))
    print('Seed points drawn!\n')
    
    # 8. Area Growing
    print('Performing tree seed growing...\n')
    L = area_growing(seeds, smoothed_cmm, args.th, args.thStep, args.thmin)
    print('Growing complete! Drawing...\n')
    
    # 9. Draw Grown Trees
    grown_trees_on_image(L, smoothed_cmm, save_path = os.path.join(h,'grown_trees.png'))
    print('Complete!\n')
