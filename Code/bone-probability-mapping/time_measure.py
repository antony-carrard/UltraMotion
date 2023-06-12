import matplotlib.pyplot as plt
import numpy as np
import math
import time
import cv2

# Start the timer
start = time.time()

# Open the image
img_path = "../../Data/Images_set_resized//3DUS_L_probe1_conf1_ds1.dcm_p350.jpg"
img = cv2.imread(img_path, 0)

# Apply a Gaussian filter
ksize = 15
blur = cv2.GaussianBlur(img,(ksize,ksize),0)

# Apply a threshold on the image
threshold = 0.2
ret,thresh = cv2.threshold(blur,round(threshold*255),1,cv2.THRESH_BINARY)

# Remove the top layer
proportion = 0.2
height = round(proportion * img.shape[0])
thresh[:height] = 0
thresh = thresh.astype(np.float64)

# Apply a LoG
laplacian = cv2.Laplacian(blur, ddepth=cv2.CV_16S, ksize=5)

# Only keep the negative pixels
laplacian = -laplacian
laplacian = laplacian / np.max(laplacian)
laplacian = np.clip(laplacian, 0, 1)

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

shadow_image = shadow_value(img, sigma=20)
shadow_image = shadow_image / np.max(shadow_image)
shadow_image = 1 - shadow_image

# Apply a Log-Gabor filter
number_scales = 5          # scale resolution
number_orientations = 8    # orientation resolution
N = 256                    # image dimensions
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
            
# Generate the filter bank
filters = get_filter_bank(number_scales, number_orientations)

def apply_log_gabor_filter_conv(img, filter):
    ifft_filter = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(filter)))
    return cv2.filter2D(img,-1,ifft_filter.real) + cv2.filter2D(img,-1,ifft_filter.imag)*1j

def normalize(img):
    img.real = (img.real - np.amin(img.real)) / (np.amax(img.real) - np.amin(img.real))
    img.imag = (img.imag - np.amin(img.imag)) / (np.amax(img.imag) - np.amin(img.imag))
    return img

log_gabor_img = []
log_gabor_img_summed = np.zeros(img.shape, dtype=complex)
for no in range(number_orientations):
    for ns in range(number_scales):
        log_gabor_img.append(apply_log_gabor_filter_conv(img, filters[no*number_scales+ns][0]))
        log_gabor_img_summed += log_gabor_img[-1]

# Rescale the image
log_gabor_img_summed = normalize(log_gabor_img_summed)

# Apply the integrated backscattering (IBS)
def IBS(img):
    squared_image = np.square(img)
    
    # Cumulative sum of each row
    ibs = np.cumsum(squared_image, axis=0)
        
    # Normalize between 0 and 1
    ibs = ibs / np.max(ibs)
    
    return ibs

ibs = IBS(img)

# Compute the final result
prob_map = laplacian * thresh * shadow_image * log_gabor_img_summed.real * log_gabor_img_summed.imag * ibs
prob_map /= np.max(prob_map)

# Find the top segment
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

# Only keep the top pixel of each column of the biggest blob
segment = np.zeros(prob_map.shape)
for idx, column in enumerate(out.T):
    top_pixel = np.where(column == 255)[0]
    if len(top_pixel) > 0:
        segment[top_pixel[0], idx] = 255

# Trace the segment on the original image
image_with_segment = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
image_with_segment[segment==255] = [255, 0, 0]

finish = time.time()

# Display the image
fig, ax = plt.subplots(1,3, figsize=(8, 6))
fig.suptitle(f'Image generated in {finish-start:.3f} s')
ax[0].imshow(img, cmap='gray')
ax[0].set_title('original image')
ax[0].axis('off')
ax[1].imshow(prob_map, cmap='gray')
ax[1].set_title('Probability map')
ax[1].axis('off')
ax[2].imshow(image_with_segment, cmap='gray')
ax[2].set_title('Image with segment')
ax[2].axis('off')
# ax[2].imshow(prob_map_thresh, cmap='gray')
# ax[2].set_title('Final result binarized')
# ax[2].axis('off')
plt.tight_layout()
plt.show()