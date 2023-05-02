# -*- coding: utf-8 -*-
"""
Created on Tue May  2 11:52:38 2023

@author: Liam
"""

import numpy as np
from tqdm import tqdm
import gdal
import matplotlib.pyplot as plt
from math import trunc

#%% Create DTM
def create_dtm(pointcloud, resolution, save_path = None, show_im = True, method = 'nearest'):
    '''Creates a dtm, interpolating the points via 'method'
    Requires pointcloud to be classified (classes 2 or 9 used here, which are ASPRS 'Ground' and 'Water')
    Shows the image if desired
    Saves to tif (unprojected) 
    
    TODO: Project the tif
    TODO: Give resolution physical meaning (can we define it in metres?)
    
    Arguments:
        pointcloud: np array. The pointcloud to create the dtm. Must be classified, and transformed to geospatial (see project_las_geospatial())
        resolution: The resolution of the DTM. 
        save_path: str. Path to save the DTM image to
        show_im: Whether to plot the final DTM
        method: interpolation method (nearest, linear, cubic)
    
    Returns:
        dtm: a numpy array of the dtm
        Saves the DTM as a tif (unprojected)
    
    '''
    # Format the points and their values for use in the interpolation function
    w = trunc((pointcloud[:,0].max() - pointcloud[:,0].min()) * resolution)
    h = trunc((pointcloud[:,1].max() - pointcloud[:,1].min()) * resolution)
    shape = (w, h)
    print(f'Created DTM will have shape {shape}')
    # Filter the points to ground and water classes only
    pointcloud = pointcloud[np.logical_or(pointcloud[:,3] == 2, pointcloud[:,3] == 9)]
    # Check you have points to interpolate
    if len(pointcloud) == 0:
        raise Exception('No points of class 2 or 9 found! Is the pointcloud classified?')
    pointcloud[:,0] = (pointcloud[:,0] - pointcloud[:,0].min()) * resolution
    pointcloud[:,1] = (pointcloud[:,1] - pointcloud[:,1].min()) * resolution
    # Formatting prior to DTM generation
    points = (pointcloud[:,0], pointcloud[:,1])
    values = (pointcloud[:,2])
    grid = np.indices((shape))
    # Interpolate using 'method' (cubic, linear, nearest)
    from scipy import interpolate
    dtm = interpolate.griddata(points, values, (grid[0,:,:], grid[1,:,:]), method = method)
    dtm = np.nan_to_num(dtm)
    dtm = np.rot90(dtm,1 )
    
    # Saving image
    if save_path:
        arr = dtm
        [rows, cols] = arr.shape
        driver = gdal.GetDriverByName('GTiff')
        out_path = save_path
        outdata = driver.Create(out_path, cols, rows, 1, gdal.GDT_Float32)
        outdata.GetRasterBand(1).WriteArray(arr)
        outdata.FlushCache()
        outdata = None
        ds = None
    if show_im == True:
        plt.imshow(dtm)
        plt.title('DTM')
        plt.show()
    return dtm

#%% Create CHM
def create_chm(pointcloud, resolution, save_path = None, show_im = True, method = 'nearest'):
    '''Creates a canopy height model, interpolating the points via 'method'
    Requires pointcloud to be classified (class 5 used here, which is ASPRS 'High Vegetation')
    Shows the image if desired
    Saves to tif (unprojected) 
    
    TODO: Project the tif
    TODO: Give resolution physical meaning (can we define it in metres?)
    
    Arguments:
        pointcloud: np array. The pointcloud to create the dtm. Must be classified, and transformed to geospatial (see project_las_geospatial())
        resolution: The resolution of the DTM. 
        save_path: str. Path to save the DTM image to
        show_im: Whether to plot the final DTM
        method: interpolation method (nearest, linear, cubic)
    
    Returns:
        dtm: a numpy array of the dtm
        Saves the DTM as a tif (unprojected)
    
    '''
    # Init sizes
    from math import trunc
    w = trunc((pointcloud[:,0].max() - pointcloud[:,0].min()) * resolution)
    h = trunc((pointcloud[:,1].max() - pointcloud[:,1].min()) * resolution)
    shape = (h, w)
    print(f'Created CHM will have shape ({w}, {h})')
    chm = np.zeros((shape))
    # Filter to 'high vegetation' class
    pointcloud = pointcloud[pointcloud[:,3] == 5]
    if len(pointcloud) == 0:
        raise Exception('No points of class 5 found! Is the pointcloud classified?')
    pointcloud[:,0] = (pointcloud[:,0] - pointcloud[:,0].min()) * resolution
    pointcloud[:,1] = (pointcloud[:,1] - pointcloud[:,1].min()) * resolution
    # Here, we dont want a smooth surface so we dont interpolate.
    # We just assign point values to their locations
    xs = np.trunc(pointcloud[:,0]-1).astype(int)
    ys = np.trunc(pointcloud[:,1]-1).astype(int)
    chm[ys, xs] = pointcloud[:,2]
    chm = np.fliplr(np.rot90(chm, 2))
    # Saving image
    if save_path:
        arr = chm
        [rows, cols] = arr.shape
        driver = gdal.GetDriverByName('GTiff')
        out_path = save_path
        outdata = driver.Create(out_path, cols, rows, 1, gdal.GDT_Float32)
        outdata.GetRasterBand(1).WriteArray(arr)
        outdata.FlushCache()
        outdata = None
        ds = None
    if show_im == True:
        plt.imshow(chm)
        plt.title('CHM')
        plt.show()
    return chm

#%% Create CMM
def create_cmm(chm, dtm, show_im = True):
    '''
    Creates canopy model from canopy height model and DTM. CHM and DTM must be same shape

    Parameters
    ----------
    chm : np array
        Canopy height model (returned from create_chm())
    dtm : np array
        Digital terrain model (returned from create_dtm())
    show_im: bool, optional
        Shows the created CMM. Default is true
    
    Returns
    -------
    cmm : array
        cmm = chm - dtm

    '''
    assert chm.shape == dtm.shape, 'CHM and DTM are different shapes!'
    # Create
    cmm = chm - dtm
    # Should not have any negative points
    cmm[np.where(cmm < 0)] = 0
    
    if show_im == True:
        plt.imshow(cmm)
        plt.title('CMM')
        plt.show()
        
    return cmm

#%% Smoothing (sometimes helps)
def gaussian_filter(arr, sigma, show_im = True):
    '''
    Gaussian blurs the array. Can be useful to remove/mask noise

    Parameters
    ----------
    arr : array
        The array to blur
    sigma : size of blur
        gaussian blur size value (higher is blurrier / bigger kernel)
    show_im: bool, optional
        Shows the blurred array. Default is true

    Returns
    -------
    gaus_arr : array
        blurred array

    '''
    from scipy.ndimage import gaussian_filter
    gaus_arr = gaussian_filter(arr, sigma)
    
    if show_im == True:
        plt.imshow(gaus_arr)
        plt.title(f'Sigma = {sigma}')
        plt.show()
    
    return gaus_arr

#%% Detecting Local Maxima
def detect_local_maxima(arr, window_size = 1, min_height = 1):
    '''
    Detects local maxima in the array. Here, you will likely input the CMM or smoothed CMM.
    Essentially, finds all points that are above min_height. Then looks in a window around each point.
    If the point is the highest in the window around it (aka, a local maxima), then call it a 'seed' (tree stem).
    

    Parameters
    ----------
    arr : array
        Likely the cmm or a smoothed cmm. This is the array we are finding local maxima in
    window_size : int, optional
        The size of the search window to look around each maxima candidate. The bigger the window, the less local maxima will be found.
        But a small window is susceptible to noise. The default is 1.
    min_height : int, optional
        The minimum height to call a tree. The default is 1.

    Returns
    -------
    seeds : list
        Tree stem candidates. [[x1, y1, height1], ...]

    '''
    # Get location array
    locs = np.indices((arr.shape))
    locs = np.moveaxis(locs,0,2)
    arr = np.reshape(arr, (arr.shape[0], arr.shape[1], 1))
    # Our loc_height_arr is shape ((x1, y1, z1),(x2, y2, z2)...)
    loc_height_arr = np.concatenate((locs, arr), axis = 2)
    # Get the highest points in the array. arr_1d become sorted based on heights (ie, the highest (x,y,z) point in the array is first,
    # second highest is second, etc). Also remove any points that are shorter than min_height
    arr_1d = np.reshape(loc_height_arr, (int(loc_height_arr.size / 3), 3))
    arr_1d = arr_1d[np.where(arr_1d[:,2] > min_height)]
    arr_1d = np.flipud(arr_1d[np.argsort(arr_1d[:,2])])
    seeds = []
    # Init progress bar
    num_pts = len(arr_1d)
    pbar = tqdm(total = num_pts)

    # Iterate over every possible seed point
    for point in arr_1d:
        # Get the point (x,y), and get the height at that point
        pt = point[0:2]
        height = point[2]
        # Get window around the point. In this window, we check for any other 'high points'
        y = int(pt[0])
        x = int(pt[1])
        window_left = max(x - window_size, 0)
        window_bottom = max(y - window_size, 0)
        window_right = x + window_size + 1
        window_top = y + window_size + 1
        window = arr[window_bottom:window_top, window_left:window_right]
        # If the candidate point is the highest point in the window placed around it, call it a seed!
        if height >= window.max():
            seeds.append([x,y,height])
        pbar.update(1)
    print(f'\n{len(seeds)} seeds found!')
    return seeds

#%% Drawing seeds on image
def seed_pts_on_image(seed_pts: list, background_arr, cmm, save_path = 'seeds.png'):
    '''
    Draws the seed points on the background array as red dots. Saves to save_path.
    Currently, background array must be a 1 channel 2D array (ie, not an image)
    
    TODO: Modify to allow the passing of images rather than just arrays

    Parameters
    ----------
    seed_pts : list
        The seed points. Likely returned from detect_local_maxima(). [[x1, y1, height1], ...]
    background_arr : arr
        Array to draw seed points on
    cmm: the cmm returned from above
    save_path : str, optional
        Where to save the PNG. Must end in png. The default is 'seeds.png'.

    Returns
    -------
    Saves the image to save_path

    '''
    assert save_path.endswith('.png'), 'save_path must end in .png!'
    assert len(background_arr.shape) == 2, 'background array must be a 1 channel 2D array (ie, not an image)'
    from PIL import Image, ImageDraw
    # Apply heatmap to background array
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin = background_arr.min(), vmax = background_arr.max())
    bg_img = cmap(norm(background_arr))
    bg_img = bg_img * 255
    bg_img = bg_img.astype('uint8')
    bg_img = Image.fromarray(bg_img, mode='RGBA')
    
    # Create seed point array (array where seed points are set to a tree number. Elsewhere is 0)
    seed_arr = np.zeros((cmm.shape), dtype = int)
    tree_num = 1
    for tree in seed_pts:
        x = tree[0]
        y = tree[1]
        seed_arr[y,x] = tree_num
        tree_num += 1
    
    # Take seed array and draw dots on the background array where there is a seed
    seed_list = np.array(np.where(seed_arr != 0)).T
    draw = ImageDraw.Draw(bg_img)
    for seed in seed_list:
        x = seed[0]
        y = seed[1]
        draw.point((y, x), fill = 'red')
    
    # Save
    bg_img.save(save_path,'PNG')

#%% Area Growing
def _grav_center(L, tree_id):
    '''
    Calculates the 'gravity center' of a tree (tree_id) in the array. 
    L is an array where cells are either 0 (no tree) or an integer (a tree number).
    'Gravity centre' is like the mean location of the tree
    

    Parameters
    ----------
    L : array
        L is an array where cells are either 0 (no tree) or an integer (a tree number).
    tree_id : int
        The ID integer of the tree you want to find the center of

    Returns
    -------
    x_mean : float
        x coordinate of the gravity center of the tree.
    y_mean : float
        y coordinate of the gravity center of the tree.

    '''
    locs = np.where(L == tree_id)
    y_mean = np.mean(locs[0])
    x_mean = np.mean(locs[1])
    return x_mean, y_mean

def _euc_dist(pt1, pt2):
    '''
    Calcualtes the euclidean distanec between two points

    Parameters
    ----------
    pt1 : tuple
        (x1,y2)
    pt2 : tuple
        (x2,y2)

    Returns
    -------
    dist : float
        Euc distance between pt1 and pt2

    '''
    x1 = pt1[0]
    y1 = pt1[1]
    x2 = pt2[0]
    y2 = pt2[1]
    dist = ((x2-x1)**2 + (y2-y1)**2)**(1/2)
    return dist

def area_growing(seed_pts, cmm, th = 28, thStep = 0.5, thmin = 1):
    '''
    From: A Segmentation-Based Method to Retrieve Stem Volume Estimates from 3-D Tree Height Models Produced by Laser Scanners

    Parameters
    ----------
    seed_pts : list
        The seed points. Likely returned from detect_local_maxima(). [[x1, y1, height1], ...]
    cmm : arr
        The CMM. Likely returned from create_cmm, or is smoothed with gaussian
    th:
        Initial threshold
    thStep:
        threshold step
    thmin:
        minimum threshold

    Returns
    -------
    L : array
        Same size as cmm. Cells are assigned 'tree ids'. All cells with the same integer number in them are considered
        to be part of one tree. Zero where there are no trees

    '''
    # Create seed point array (array where seed points are set to a tree number. Elsewhere is 0)
    seed_arr = np.zeros((cmm.shape), dtype = int)
    tree_num = 1
    for tree in seed_pts:
        x = tree[0]
        y = tree[1]
        seed_arr[y,x] = tree_num
        tree_num += 1
    
    th = 28
    thStep = 0.5
    thmin = 1
    # Zero pad CMM
    cmm = np.pad(cmm, 1, 'constant', constant_values = (0))
    # 1. Select Seed Pts (done - it's an input)
    # 2. Initialize L
    L = np.copy(seed_arr)
    # 3. Begin loop. Decrease the threshold. If minimum threshold reached, stop
    pbar = tqdm(total = (th-thmin)/thStep)
    while True:
        Q = list(np.array(np.where(L != 0)).T.tolist())
        pbar.update(1)
        th = th - thStep
        if th < thmin:
            break
        # 4. Take the next pixel from Q. If this is the end of the Q, go to step 3
        while True:
            if len(Q) > 0:
                seed = Q.pop(0)
                seed_num = L[seed[0], seed[1]]
                x_cent, y_cent = _grav_center(L, seed_num)
                tree_pts = np.array((np.where(L == seed_num)[0], np.where(L == seed_num)[1])).T
                max_dist = 0
                tree_pts[:,0] = (tree_pts[:,0] - x_cent)**2
                tree_pts[:,1] = (tree_pts[:,1] - y_cent)**2
                dists = np.array(((tree_pts[:,0] + tree_pts[:,1])**(1/2)))
                max_dist = dists.max()
            else:
                break
            # 5. Select the neighbour pixel. If there are no neighbours left, go to 8
            neighbours = [[seed[0]-1, seed[1]], [seed[0], seed[1]-1], [seed[0]+1, seed[1]], [seed[0], seed[1]+1]]
            # print(neighbours)
            for neighbour in neighbours:
                n_y = neighbour[0]
                n_x = neighbour[1]
                # 6. If I[neighbour] > th, add pixel to Q and mark the label in L
                dist_n_grav = _euc_dist((x_cent, y_cent), (n_x, n_y))
                if (cmm[n_y+1, n_x+1] > th) and (dist_n_grav < max_dist+1):
                    # Only mark if L is uncommitted
                    if L[n_y, n_x] == 0:
                        Q.append(neighbour)
                        L[n_y, n_x] = seed_num
                # 7. Go to 5 (restart the loop)
            # 8. If the pixel has no more uncomitted neighbours, remove the pixel from Q (achieved with pop)
            # 9. Go to step 4
    return L

#%% Illustrate the grown area
def grown_trees_on_image(L, background_arr, save_path = 'grown_trees.png'):
    '''
    Draws the grown trees on the background array. Saves to save_path.
    Currently, background array must be a 1 channel 2D array (ie, not an image)
    
    TODO: Modify to allow the passing of images rather than just arrays

    Parameters
    ----------
    L : array
        The tree array. Likely returned from area_growing()
    background_arr : arr
        Array to draw trees on
    save_path : str, optional
        Where to save the PNG. Must end in png. The default is 'grown_trees.png'.

    Returns
    -------
    Saves the image to save_path

    '''
    from PIL import Image, ImageDraw
    import random
    assert save_path.endswith('.png'), 'save_path must end in .png!'
    assert len(background_arr.shape) == 2, 'background array must be a 1 channel 2D array (ie, not an image)'
    # Apply heatmap to the background image
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin = background_arr.min(), vmax = background_arr.max())
    bg_img = cmap(norm(background_arr))
    bg_img = bg_img * 255
    bg_img = bg_img.astype('uint8')
    bg_img = Image.fromarray(bg_img, mode='RGBA')
    draw = ImageDraw.Draw(bg_img)
    
    # Get every tree id in the array
    tree_nums = set(L[np.nonzero(L)])
    # Iter through
    for tree_num in tree_nums:
        # Apply a random colour to each tree
        tree = np.array(np.where(L == tree_num)).T
        fill = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        for pt in tree:
            x = pt[0]
            y = pt[1]
            draw.point((y,x), fill = fill)
    # Save
    bg_img.save(save_path,'PNG')