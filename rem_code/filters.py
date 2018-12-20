# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 11:28:47 2017

@author: Karthick
"""
import numpy as np

def combine_img(band_r,band_g,band_b):
    """Combine r ,g ,b to produce a colour image for tiling"""

    rgb_img = np.append([band_r.flatten()], [band_g.flatten()], axis=0)
    rgb_img = np.transpose(np.append(rgb_img, [band_b.flatten()], axis=0))
    rgb_img_reshape = rgb_img.reshape(band_r.shape[1],band_r.shape[2],3)
    return rgb_img_reshape

def convert_leading_channel(rgb_img):
    rgb_img_reshape = rgb_img.reshape(rgb_img.shape[1],rgb_img.shape[2],3)
    return rgb_img_reshape

def apply_threshold_filter(rastered_tif,threshold):
    """Apply thresholding to the image"""

    new_img = np.asarray(rastered_tif.copy())
    new_img[new_img<threshold] = 0
    new_img[new_img>=threshold] = 1
    return new_img



def apply_MSR(b2,b3,b4,b5):
    """Apply Multiband spectral relationship to the image"""
    #(b2 + b3) > (b4 + b5)

    return np.uint8(((np.add(b2,b3) > np.add(b4,b5))))


def apply_NDVI(b2,b3,b4,b5):
    """Apply NDVI water index to the image"""
    # (b4 – b3)/(b4 + b3)

    return (np.subtract(b4,b3) / np.add(b4, b3))


def apply_NDWI(b2,b3,b4,b5):
    """Apply Multiband spectral relationship to the image"""
    # (b2 – b4)/(b2 + b4)

    return (np.subtract(b2,b4) / np.add(b2, b4))

def apply_MNDWI(b2,b3,b4,b5):
    """Apply Multiband spectral relationship to the image"""
    # (b2 – b5)/(b2 + b5)

    return (np.subtract(b2,b5) / np.add(b2, b5))

def apply_NDBI(b2,b3,b4,b5):
    """Apply Multiband spectral relationship to the image"""
    # (b5 – b4)/(b5 + b4)

    return (np.subtract(b5,b4) / np.add(b5,b4))

