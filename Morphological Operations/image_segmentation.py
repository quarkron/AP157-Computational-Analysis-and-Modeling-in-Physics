import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from PIL import Image as im
import cv2  
from skimage import filters  

#Convert to normalized chromaticity coordinates
def ncc(RGB):
    I = np.sum(RGB,axis=2) 
    I[I==0] = 1000000
    r,g= (RGB[:,:,0]/I), (RGB[:,:,1]/I) 
    b = 1-(r+g)
    result = np.stack([r,g,b],axis=2) 
    result[result==0] = 100000000
    return np.stack([r,g,b],axis=2) # Stack R, G, and B arrays along the third dimension


#Plots only the r and g channels in the NCC of the image.
def ncc_plotter(image, size=(10,10), color=True):
    image_ncc = ncc(image)
    r,g = image_ncc[:,:,0], image_ncc[:,:,1]
    b=1-(r+g)
    RGB_flat = image_ncc.reshape(-1,3)
    plt.figure(figsize=size)
    if color:
        plt.scatter(r,g, c=RGB_flat)
    else: 
        plt.scatter(r,g, c='k')
    plt.xlim([0,1])
    plt.ylim([0,1]) 

def ROI(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    r_coords = cv2.selectROI(image)
    roi = image[int(r_coords[1]):int(r_coords[1]+r_coords[3]),
                int(r_coords[0]):int(r_coords[0]+r_coords[2])]
    ROI = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    return ROI 

def gaussian(mu,sigma, x): 
    mu,sigma,x= np.array(mu), np.array(sigma), np.array(x)
    if sigma == 0:
        a = np.zeros_like(x) 
        b=a
    else:
        a = (1/(sigma*np.sqrt(2*np.pi))) 
        b = np.exp(-(x-mu)**2/(2*sigma**2)) 
    return a*b 

def gaussian_segmenter(image, binarized=True):
    ROI_image = ROI(image)
    ROI_ncc = ncc(ROI_image)
    image_ncc = ncc(image)
    r_mean, g_mean = np.mean(ROI_ncc[:,:,0]), np.mean(ROI_ncc[:,:,1])
    r_std, g_std = np.std(ROI_ncc[:,:,0]), np.std(ROI_ncc[:,:,1])
    P = gaussian(r_mean, r_std,image_ncc[:,:,0])*gaussian(g_mean, g_std, image_ncc[:,:,1])
    if binarized: 
        threshold_value = filters.threshold_otsu(P[np.isfinite(P)])   #handle only finite values, remove NaNs 
        P = P > threshold_value
    return P 

def colored_gaussian_segmenter(image): 
    P = gaussian_segmenter(image,binarized=True)
    segmented_image = np.stack([P*image[:,:,0], P*image[:,:,1], P*image[:,:,2]], axis=2)
    return segmented_image 

def nonparametric_segmenter(image, bins=32, binarized=True):
    ROI_image = ROI(image)
    ROI_ncc, image_ncc = ncc(ROI_image), ncc(image) 

    #2D histogram of ROI
    x, y= ROI_ncc[:,:,0].flatten(), ROI_ncc[:,:,1].flatten()
    H = np.histogram2d(y,x, bins=bins, range=[[0,1],[0,1]])

    #Vectorized look-up algorithm based on 2D histogram
    x_indices = np.digitize(image_ncc[:,:,0], H[1]) - 1
    y_indices = np.digitize(image_ncc[:,:,1], H[2]) - 1

    # Prevent possible IndexErrors by clipping the values to the valid index ranges.
    x_indices = np.clip(x_indices, 0, bins-1)
    y_indices = np.clip(y_indices, 0, bins-1)
    
    P = H[0][y_indices, x_indices]
    if binarized: 
        threshold_value = filters.threshold_otsu(P[np.isfinite(P)])   #handle only finite values, remove NaNs 
        P = P > threshold_value
    return P
