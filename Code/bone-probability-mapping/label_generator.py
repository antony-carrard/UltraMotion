import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import os

DIM = 256

# Open all the images in the folder
import glob
image_list = []
path = '..\\..\\Data\\Images_set_resized\\*.jpg'
filenames = glob.glob(path)
for filename in filenames:
    # print(filename)
    img = cv2.imread(filename, 0)
    image_list.append(img)

number_of_images = len(image_list)
print(f'All images opened. Number of images: {number_of_images}')

# Gaussian filter
gaussian_list = []
ksize = 25
for img in image_list:
    gaussian_list.append(cv2.GaussianBlur(img,(ksize,ksize),0))
print(f'Gaussian filter applied')

# Apply a threshold on the blurred image
threshold_list = []
threshold = 0.2
proportion = 0.2
for img in gaussian_list:
    ret,thresh = cv2.threshold(img,round(threshold*255),1,cv2.THRESH_BINARY)
    height = round(proportion * img.shape[0])
    thresh[:height] = 0
    thresh = thresh.astype(np.float64)
    threshold_list.append(thresh)
print(f'Threshold applied')

# Apply the Laplacian on Gaussian (LoG)
ksize=5
laplacian_list = []
for img in gaussian_list:
    laplacian = cv2.Laplacian(img, ddepth=cv2.CV_16S, ksize=ksize)

    # Only keep the negative pixels
    laplacian = -laplacian
    laplacian = laplacian / np.max(laplacian)
    laplacian = np.clip(laplacian, 0, 1)
    laplacian_list.append(laplacian)
print(f'LoG applied')

# Apply the shadow model
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def shadow_value(img, mu=0, sigma=40):
    shadow_img = np.zeros(img.shape)
    R, C = img.shape
    G = np.tile(gaussian(np.arange(R), mu, sigma), (C, 1)).T
    for a in range(R):
        I = img[a:]
        # Compute the numerator and denominator of the shadow formula
        numerator = np.sum(G[:R-a] * I, axis=0)
        denominator = np.sum(G[:R-a], axis=0)
        
        # Compute the shadow value and return it
        shadow_line = numerator / denominator
        shadow_img[a] = shadow_line
                
    return shadow_img

sigma=50
shadow_list = []
print_each = 10
for idx, img in enumerate(gaussian_list):
    shadow_image = shadow_value(img, sigma=sigma)
    shadow_image = shadow_image / np.max(shadow_image)
    shadow_image = 1 - shadow_image
    shadow_list.append(shadow_image)
    if idx % print_each==0: print(f'Shadow model applied on {idx} images')
print(f'Shadow model applied')

# Apply the Log-Gabor filter
number_scales = 6          # scale resolution
number_orientations = 6    # orientation resolution
N = DIM                    # image dimensions
def getFilter(f_0, theta_0):
    # filter configuration
    scale_bandwidth =  0.996 * math.sqrt(2/3)
    angle_bandwidth =  0.996 * (1/math.sqrt(2)) * (np.pi/number_orientations)

    # x,y grid
    extent = np.arange(-N/2, N/2 + N%2)
    x, y = np.meshgrid(extent,extent)

    mid = int(N/2)
    ## orientation component ##
    theta = np.arctan2(y,x)
    center_angle = ((np.pi/number_orientations) * theta_0) if (f_0 % 2) \
                else ((np.pi/number_orientations) * (theta_0+0.5))

    # calculate (theta-center_theta), we calculate cos(theta-center_theta) 
    # and sin(theta-center_theta) then use atan to get the required value,
    # this way we can eliminate the angular distance wrap around problem
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    ds = sintheta * math.cos(center_angle) - costheta * math.sin(center_angle)    
    dc = costheta * math.cos(center_angle) + sintheta * math.sin(center_angle)  
    dtheta = np.arctan2(ds,dc)

    orientation_component =  np.exp(-0.5 * (dtheta/angle_bandwidth)**2)

    ## frequency componenet ##
    # go to polar space
    raw = np.sqrt(x**2+y**2)
    # set origin to 1 as in the log space zero is not defined
    raw[mid,mid] = 1
    # go to log space
    raw = np.log2(raw)

    center_scale = math.log2(N) - f_0
    draw = raw-center_scale
    frequency_component = np.exp(-0.5 * (draw/ scale_bandwidth)**2)

    # reset origin to zero (not needed as it is already 0?)
    frequency_component[mid,mid] = 0

    return frequency_component * orientation_component, frequency_component, orientation_component

def get_filter_bank(number_scales, number_orientations):
    filter_bank = []
    
    # Loop over orientations and scales
    for o in range(number_orientations):
        for s in range(number_scales):
            
            filter_bank.append(getFilter(s+1, o))
            
    return filter_bank

def apply_log_gabor_filter(img, filter):
    ifft_filter = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(filter)))
    return cv2.filter2D(img,-1,ifft_filter.real) + cv2.filter2D(img,-1,ifft_filter.imag)*1j

def normalize(img):
    img.real = (img.real - np.amin(img.real)) / (np.amax(img.real) - np.amin(img.real))
    img.imag = (img.imag - np.amin(img.imag)) / (np.amax(img.imag) - np.amin(img.imag))
    return img

gabor_list = []
filters = get_filter_bank(number_scales, number_orientations)
print_each = 10
for idx, img in enumerate(image_list):
    log_gabor_img = []
    log_gabor_img_summed = np.zeros(img.shape, dtype=complex)
    for no in range(number_orientations):
        for ns in range(number_scales):
            log_gabor_img.append(apply_log_gabor_filter(img, filters[no*number_scales+ns][0]))
            log_gabor_img_summed += log_gabor_img[-1]

    log_gabor_img_summed = normalize(log_gabor_img_summed)
    gabor_list.append(log_gabor_img_summed)
    if idx % print_each==0: print(f'Log Gabor filter applied on {idx} images')
print(f'Log Gabor filter applied')

# Apply the integrated backscattering
def IBS(img):
    squared_image = np.square(img)
    
    # Cumulative sum of each row
    ibs = np.cumsum(squared_image, axis=0)
        
    # Normalize between 0 and 1
    ibs = ibs / np.max(ibs)
    
    return ibs

ibs_list = []
for img in image_list:
    ibs_list.append(IBS(img))
print(f'IBS applied')

# Compute the final probability map
final_map_list = []
for i in range(number_of_images):
    prob_map = laplacian_list[i] * threshold_list[i] * shadow_list[i] * gabor_list[i].real* gabor_list[i].imag * ibs_list[i]
    prob_map /= np.max(prob_map)

    # Trim the values too low
    threshold = 0.08
    threshold_int = round(threshold*255)

    # ret,img_uint = cv2.threshold(prob_map*255,threshold_int,255,cv2.THRESH_BINARY)
    img_uint = (prob_map*255).astype(np.uint8)

    # Generate intermediate image; use morphological closing to keep parts of the bone together
    str_elem_size = 5
    inter = cv2.morphologyEx(img_uint, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (str_elem_size, str_elem_size)))

    ret,inter_thresh = cv2.threshold(inter,threshold_int,255,cv2.THRESH_BINARY)
    inter_thresh = inter_thresh.astype(np.uint8)
    # inter = cv2.bitwise_not(inter)

    # Find largest contour in intermediate image
    cnts, _ = cv2.findContours(inter_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Output
    out = np.zeros(img_uint.shape, np.uint8)
    if len(cnts) > 0:
        cnt = max(cnts, key=cv2.contourArea)
        cv2.drawContours(out, [cnt], -1, 255, cv2.FILLED)

    final_map_list.append(out)
print(f'Final map computed')

# Write the images and their labels
dir_name = '..\\..\\Data\\labeled_images'
for (img, filename) in zip(final_map_list, filenames):
    img_name = os.path.basename(filename)
    full_path = os.path.join(dir_name, img_name + '.' + 'png')
    cv2.imwrite(full_path, img)
print(f'Image saved')

print('Image labelling done')