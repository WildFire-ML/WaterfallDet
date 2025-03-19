# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 09:45:20 2024

@author: Labadmin
"""
import laspy as lp
import numpy as np


def load_las(las_path):
    """Loads the las file at las_path. For class definitions, find "ASPRS LAS SPECIFICATION 1.4"

    Args:
        las_path (str): path to the las file

    Returns:
        las_data (np.array): points as [X, Y, Z, Class]
        las_transform_dict (dict): transformation parameters of points for LAS coords to geospatial coords
    """
    # Backwards compatability support
    if lp.__version__.startswith('1.'):
        inFile = lp.file.File(las_path, mode='r')
        las_data = np.vstack([inFile.X, inFile.Y, inFile.Z, inFile.Classification]).transpose()
    elif lp.__version__.startswith('2.'):
        inFile = lp.read(las_path)
        las_data = np.vstack([inFile.X, inFile.Y, inFile.Z, inFile.classification]).transpose()
    scalex = inFile.header.scale[0]
    offsetx = inFile.header.offset[0]
    scaley = inFile.header.scale[1]
    offsety = inFile.header.offset[1]
    scalez = inFile.header.scale[2]
    offsetz = inFile.header.offset[2]
    if lp.__version__.startswith('1.'):
        inFile.close()

    las_transform_dict = {'scalex': scalex, 'offsetx': offsetx, 'scaley': scaley,
                          'offsety': offsety, 'scalez': scalez, 'offsetz': offsetz}

    return las_data, las_transform_dict


def project_las_geospatial(las_data, las_transform_dict):
    """Converts las points from las coordinates to geospatial coordinates

    Args:
        las_data (np.array): points as [X, Y, Z, Class] as returned by load_las
        las_transform_dict (dict): transformation parameters of points for LAS coords to geospatial coords

    Returns:
        transformed_las_data (np.array): points as [X, Y, Z, Class] in geospatial coords
    """
    # Convert las data to float
    transformed_las_data = las_data.astype('float')
    # Transform in x, y, z
    transformed_las_data[:, 0] = ((las_data[:, 0] * las_transform_dict['scalex']) + las_transform_dict['offsetx'])
    transformed_las_data[:, 1] = ((las_data[:, 1] * las_transform_dict['scaley']) + las_transform_dict['offsety'])
    transformed_las_data[:, 2] = ((las_data[:, 2] * las_transform_dict['scalez']) + las_transform_dict['offsetz'])

    return transformed_las_data
