# Import the necessary libraries

import numpy as np
from sewar import full_ref
from skimage import measure, metrics
from skimage import color
from skimage import io
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import glob

# LET'S RESIZE AND DISPLAY A SINGLE IMAGE

img_path = "images/clean.png"
im1= Image.open(img_path)
print("{}".format(im1.format))
print("size: {}".format(im1.size))
print("image mode: {}".format(im1.mode))
plt.imshow(im1)


#LET'S RESIZE ALL THE IMAGES IN THE FOLDER AND SAVE THEM TO A NEW FOLDER

# =============================================================================
# image_list = []
# resized_list = []
# 
# 
# for filename in glob.glob("/home/bakary/Desktop/Image_Quality_Assessment/images/*.ppm"):
#     print(filename)
#     img= Image.open(filename)
#     image_list.append(img)
# 
# 
# 
# for image in image_list:
#     #image.show()
#     image = image.resize((2500,2500))
#     resized_list.append(image)
# 
# 
# 
# # =============================================================================
# #Save the resize images ina a new folder
# for (i, new) in enumerate(resized_list):
#   new.save("{}{}{}".format("/home/bakary/Desktop/Image_Quality_Assessment/resized_images/image", i+1, " .ppm"))
# =============================================================================
# 
#=======================================================================================================
#=======================================================================================================








# LET'S LOAD THE RESIZE IMAGE AND COMPUTE ITS QUALITY OF USING VARIOUS STATISTICAL METRICES

# ==================================================================================================
#Loading the trusted image
realimg = "/home/bakary/Desktop/Image_Quality_Assessment/resized_images/image3 .ppm"
# 
real= io.imread(realimg)[:,:,:3]
rl = color.rgb2gray(real)
# plt.imshow(rl, cmap="gray")
#plt.show(real)
# # =============================================================================
# 
# Loading the ambiguous image
realimg1 = "/home/bakary/Desktop/Image_Quality_Assessment/resized_images/image1 .ppm"
# 
real1= io.imread(realimg1)[:,:,:3]
rl1 = color.rgb2gray(real1)
# plt.imshow(rl1, cmap="gray")
# plt.show()


# Computing the Mean Square Error
mse_skimg = metrics.mean_squared_error(rl, rl1)
# 
print("MSE: based on scikit-image = ", mse_skimg)

#=========================================================================================================
# Computing the Peak signal to noise ratio

# Same as PSNR available in sewar
# Older versions of skimage: skimage.measure.compare_psnr
# skimage.metrics.peak_signal_noise_ratio

psnr_skimg = metrics.peak_signal_noise_ratio(rl, rl1, data_range=None)
print("PSNR: based on scikit-image = ", psnr_skimg)

#=========================================================================================================


#Normalized root mean squared error

#Older versions of skimage: skimage.measure.compare_nrmse
rmse_skimg = metrics.normalized_root_mse(rl, rl1)
print("RMSE: based on scikit-image = ", rmse_skimg)

#=========================================================================================================

# Computing the Structral Similarity Index
from skimage.metrics import structural_similarity as ssim
ssim_skimg = ssim(rl, rl1,
                  data_range = rl1.max() - rl1.min() , 
                  channel_axis = False)
print("SSIM: based on scikit-image = ", ssim_skimg)

#==========================================================================================================

#ERGAS Global relative error
"""calculates global relative error 
GT: first (original) input image.
P: second (deformed) input image.
r: ratio of high resolution to low resolution (default=4).
ws: sliding window size (default = 8).
	:returns:  float -- ergas value.
	"""
ergas_img = full_ref.ergas(rl, rl1, r=4, ws=8)
print("EGRAS: global relative error = ", ergas_img)


####################################################################

####################################################################
#Multiscale structural similarity index
"""calculates multi-scale structural similarity index (ms-ssim).
	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:param weights: weights for each scale (default = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).
	:param ws: sliding window size (default = 11).
	:param K1: First constant for SSIM (default = 0.01).
	:param K2: Second constant for SSIM (default = 0.03).
	:param MAX: Maximum value of datarange (if None, MAX is calculated using image dtype).
	:returns:  float -- ms-ssim value.
	"""
msssim_img=full_ref.msssim(rl, rl1, weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333], ws=11, 
                           K1=0.01, K2=0.03, MAX=255)

print("MSSSIM: multi-scale structural similarity index = ", msssim_img)


##############################################################################

#PSNR
"""calculates peak signal-to-noise ratio (psnr).
	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:param MAX: maximum value of datarange (if None, MAX is calculated using image dtype).
	:returns:  float -- psnr value in dB.
	"""
psnr_img=full_ref.psnr(rl, rl1, MAX=255)

print("PSNR: peak signal-to-noise ratio = ", psnr_img)

#======================================================================================================

#relative average spectral error (rase)
"""calculates relative average spectral error (rase).
	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:param ws: sliding window size (default = 8).
	:returns:  float -- rase value.
	"""
RASE_img = full_ref.rase(rl, rl1, ws=8)
#print("RASE: relative average spectral error = ", RASE_img)

#=======================================================================================================


######################################################################
#RMSE
"""calculates root mean squared error (rmse).
	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:returns:  float -- rmse value.
	"""
rmse_img = full_ref.rmse(rl, rl1)
print("RMSE: root mean squared error = ", rmse_img)


#########################################################################
#calculates spectral angle mapper (sam).
"""calculates spectral angle mapper (sam).
	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:returns:  float -- sam value.
	"""
ref_sam_img = full_ref.sam(rl, rl1)
print("REF_SAM: spectral angle mapper = ", ref_sam_img)


######################################################################


#Spatial correlation coefficient
#full_ref.scc(ref_img, img, win=[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], ws=8)

#Structural similarity index
"""calculates structural similarity index (ssim).
	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:param ws: sliding window size (default = 8).
	:param K1: First constant for SSIM (default = 0.01).
	:param K2: Second constant for SSIM (default = 0.03).
	:param MAX: Maximum value of datarange (if None, MAX is calculated using image dtype).
	:returns:  tuple -- ssim value, cs value.
	"""
ssim_img = full_ref.ssim(rl, rl1, ws=11, K1=0.01, K2=0.03, MAX=255, fltr_specs=None, mode='valid')
print("SSIM: structural similarity index = ", ssim_img)

##############################################################################

#Universal image quality index
"""calculates universal image quality index (uqi).
	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:param ws: sliding window size (default = 8).
	:returns:  float -- uqi value.
	"""
UQI_img = full_ref.uqi(rl, rl1, ws=8)
print("UQI: universal image quality index = ", UQI_img)

##############################################################################


#Pixel Based Visual Information Fidelity (vif-p)
"""calculates Pixel Based Visual Information Fidelity (vif-p).
	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:param sigma_nsq: variance of the visual noise (default = 2)
	:returns:  float -- vif-p value.
	"""
VIFP_img = full_ref.vifp(rl, rl1, sigma_nsq=2)
print("VIFP: Pixel Based Visual Information Fidelity = ", VIFP_img)

##############################################################################

















#rl=cv2.resize(rl, rl1.shape)
# 
# print (rl1.shape)
# print (rl.shape)


# realimg2 = Image.open(r"/home/bakary/Desktop/Image_Quality_Assessment/images/image2.ppm")
# # real1= io.imread(realimg1)[:,:,:3]
# # rl = color.rgb2gray(real1)
# # plt.imshow(realimg2, cmap="BuGn")
# # plt.show()
# new_image2 = realimg2.resize((225, 225))
# plt.imshow(new_image2, cmap="BuGn")
# plt.show()
# print(new_image2.size)
# 


