#!/usr/bin/python3

import numpy as np


def my_conv2d_freq(image: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """
    Apply the Convolution Theorem to perform the convolution operation. 
    
    Return 
    - the input image represented in the frequency domain, 
    - the filter represented in the frequency domain,
    - the result of the convolution in the frequency domain, and 
    - the result of the convolution in the spatial domain.
    We will plot and analyze these to gain a better understanding of what is going on.

    Args:
        image: array of shape (m, n)
        filter: array of shape (k, j)
    Returns:
        image_freq: array of shape (m, n)
        filter_freq: array of shape (m, n)
        conv_result_freq: array of shape (m, n)
        conv_result: array of shape (m, n)
    HINTS:
    - Pad your filter in the spatial domain. We want to retain all of the high frequencies after the FFT
    - Return only the real component of the convolution result
    - Numpy considers frequency graphs differently than we have shown them in class. Look into the 
      documentation for np.fft.fft2 for how to account for this in the output image.
    - When applying padding, only use the zero-padding method.
    """

    ############################
    ### TODO: YOUR CODE HERE ###

    img_h, img_w = image.shape
    filter_h, filter_w = filter.shape
    
    padded_filter = np.zeros_like(image, dtype=np.float64)
    
    pad_h_start = (img_h - filter_h) // 2
    pad_w_start = (img_w - filter_w) // 2
    
    padded_filter[pad_h_start:pad_h_start + filter_h, pad_w_start:pad_w_start + filter_w] = filter
    shifted_filter = np.fft.ifftshift(padded_filter)
    
    image_freq = np.fft.fft2(image)
    filter_freq = np.fft.fft2(shifted_filter)
    
    conv_result_freq = image_freq * filter_freq
    conv_result_complex = np.fft.ifft2(conv_result_freq)
    
    conv_result = np.real(conv_result_complex)

    ### END OF STUDENT CODE ####
    ############################

    return image_freq, filter_freq, conv_result_freq, conv_result 


def my_deconv2d_freq(image: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """
    Apply the Convolution Theorem to perform the convolution operation.
    
    Return 
    - the input image represented in the frequency domain, 
    - the filter represented in the frequency domain,
    - the result of the deconvolution in the frequency domain, and 
    - the result of the deconvolution in the spatial domain.

    We will plot and analyze these to gain a better understanding of what is going on.

    Args:
        image: array of shape (m, n)
        filter: array of shape (k, j)
    Returns:
        image_freq: array of shape (m, n)
        filter_freq: array of shape (m, n)
        deconv_result_freq: array of shape (m, n)
        deconv_result: array of shape (m, n)
    HINTS:
    - Pad your filter in the spatial domain. We want to retain all of the high frequencies after the FFT
    - Return only the real component of the deconvolution result
    - Numpy considers frequency graphs differently than we have shown them in class. Look into the 
      documentation for np.fft.fft2 to see what this means and to account for this in the output image.
    - When applying padding, only use the zero-padding method.
    """

    ############################
    ### TODO: YOUR CODE HERE ###
    
    img_h, img_w = image.shape
    filter_h, filter_w = filter.shape
    epsilon = 1e-10
    
    padded_filter = np.zeros_like(image, dtype=np.float64)
    pad_h_start = (img_h - filter_h) // 2
    pad_w_start = (img_w - filter_w) // 2
    
    padded_filter[pad_h_start:pad_h_start + filter_h, pad_w_start:pad_w_start + filter_w] = filter
    shifted_filter = np.fft.ifftshift(padded_filter)
    
    image_freq = np.fft.fft2(image)
    filter_freq = np.fft.fft2(shifted_filter)
    
    deconv_result_freq = image_freq / (filter_freq + epsilon)
    deconv_result_complex = np.fft.ifft2(deconv_result_freq)
    
    deconv_result = np.real(deconv_result_complex)

    ### END OF STUDENT CODE ####
    ############################

    return image_freq, filter_freq, deconv_result_freq, deconv_result





