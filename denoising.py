


import os, sys
from PIL import Image
for infile in os.listdir("./"):
    print ("file : " + infile)
    if infile[-3:] == "tif" or infile[-3:] == "bmp" :
       # print "is tif or bmp"
       outfile = infile[:-3] + "jpeg"
       im = Image.open(infile)
       print ("new filename : " + outfile)
       out = im.convert("RGB")
       out.save(outfile, "JPEG", quality=90)
       
       
       
       
#%%

from PIL import Image
from PIL import ImageEnhance
import cv2 as cv
# Opens the image file
image = cv.imread('output.jpeg',0)  
  
# shows image in image viewer
equ = cv.equalizeHist(image)
window_name = 'image'
# shwindow_name = 'image'ows updated image in image viewer
cv.imshow(window_name, image)
  
cv.waitKey(0)
cv.destroyAllWindows()      


#%%

import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('seaborn')
 
image = cv2.imread('res.jpeg')
dst = cv2.fastNlMeansDenoisingColored(image, None, 11, 6, 7, 21)
 
row, col = 1, 2
fig, axs = plt.subplots(row, col, figsize=(15, 10))
#fig.tight_layout()
axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0].set_title('original')
axs[0].set_axis_off()
axs[1].imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
axs[1].set_title('Denoised')
axs[1].set_axis_off()
plt.show()


#%%%



from skimage import io
import matplotlib.pyplot as plt

# read the image stack
img = io.imread('im2.tif')
# show the image

image = img[38,4, :,:]
plt.imshow(image,cmap='gray')
plt.axis('off')
#io.imsave('im2.png', image)
# save the image
plt.savefig('output2.tif', transparent=True, dpi=300, bbox_inches="tight", pad_inches=0.0)





#%%%




from skimage import io


import matplotlib.pyplot as plt

from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float
from skimage.util import random_noise


noisy  = io.imread('res.jpg')
#noisy = noisy[44,3]

sigma = 0.155
#noisy = random_noise(original, var=sigma**2)

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(8, 5),
                       sharex=True, sharey=True)

plt.gray()

# Estimate the average noise standard deviation across color channels.
#sigma_est = estimate_sigma(noisy, channel_axis=-1, average_sigmas=True)
# Due to clipping in random_noise, the estimate will be a bit smaller than the
# specified sigma.
#print(f'Estimated Gaussian noise standard deviation = {sigma_est}')

ax[0, 0].imshow(noisy)
ax[0, 0].axis('off')
ax[0, 0].set_title('Noisy')
ax[0, 1].imshow(denoise_tv_chambolle(noisy, weight=0.1))
ax[0, 1].axis('off')
ax[0, 1].set_title('TV')
ax[0, 2].imshow(denoise_bilateral(noisy, sigma_color=0.05, sigma_spatial=15))
ax[0, 2].axis('off')
ax[0, 2].set_title('Bilateral')
ax[0, 3].imshow(denoise_wavelet(noisy, rescale_sigma=True))
ax[0, 3].axis('off')
ax[0, 3].set_title('Wavelet denoising')

ax[1, 1].imshow(denoise_tv_chambolle(noisy, weight=0.2))
ax[1, 1].axis('off')
ax[1, 1].set_title('(more) TV')
ax[1, 2].imshow(denoise_bilateral(noisy, sigma_color=0.1, sigma_spatial=15
                ))
ax[1, 2].axis('off')
ax[1, 2].set_title('(more) Bilateral')
ax[1, 3].imshow(denoise_wavelet(noisy,
                                rescale_sigma=True))
ax[1, 3].axis('off')
ax[1, 3].set_title('Wavelet denoising\nin YCbCr colorspace')
ax[1, 0].imshow(noisy)
ax[1, 0].axis('off')
ax[1, 0].set_title('Original')

fig.tight_layout()

plt.show()

#%%%

from skimage import io


import matplotlib.pyplot as plt

from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float
from skimage.util import random_noise


noisy  = io.imread('res.jpg')
#noisy = noisy[44,3]

sigma = 0.155
#noisy = random_noise(original, var=sigma**2)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 15),
                       sharex=True, sharey=True)

plt.gray()

# Estimate the average noise standard deviation across color channels.
#sigma_est = estimate_sigma(noisy, channel_axis=-1, average_sigmas=True)
# Due to clipping in random_noise, the estimate will be a bit smaller than the
# specified sigma.
#print(f'Estimated Gaussian noise standard deviation = {sigma_est}')

ax[0].imshow(noisy)
ax[0].axis('off')
ax[0].set_title('Noisy')
ax[1].imshow(denoise_tv_chambolle(noisy, weight=0.5))
ax[1].axis('off')
ax[1].set_title('TV Denoised')


fig.tight_layout()

plt.show()


#%%%


from skimage import io
import matplotlib.pyplot as plt

# read the image stack
img = io.imread('im3.tif')
# show the image




for i in range(60):
    for j in range(7):
        image = img[i,j,:,:]
        plt.imshow(image,cmap='gray')
        plt.axis('off')
        plt.savefig('outpu'+str(i)+str(j)+'.jpg', transparent=True, dpi=300, bbox_inches="tight", pad_inches=0.0)



#%%%
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
import image_dehazer										# Load the library
import cv2
import matplotlib.pyplot as plt
from skimage import io

HazeImg = cv2.imread('outpu72.jpg')	
				# read input image -- **must be a color image**
HazeCorrectedImg, HazeMap = image_dehazer.remove_haze(HazeImg, showHazeTransmissionMap=False)		# Remove Haze

'''
HazeCorrectedImg = cv2.cvtColor(HazeCorrectedImg, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit = 5)
HazeCorrectedImg = clahe.apply(HazeCorrectedImg) + 30
'''

'''
cv2.imshow('input image', HazeImg);						# display the original hazy image
cv2.imshow('enhanced_image', denoise_tv_chambolle(HazeCorrectedImg, weight=0.1));			# display the result
cv2.imshow('HazeMap', HazeMap);							# display the HazeMap
cv2.waitKey(0)											# hold the display window
'''
#cv2.imwrite('res.jpg',HazeCorrectedImg)
denoised = denoise_tv_chambolle(HazeCorrectedImg, weight=0.8)
### user controllable parameters (with their default values):
    
plt.imshow(denoised,cmap='gray')
plt.axis('off')
    #io.imsave('im2.png', image)
    # save the image
plt.savefig('res.jpg', transparent=True, dpi=300, bbox_inches="tight", pad_inches=0.0)
   
airlightEstimation_windowSze=15
boundaryConstraint_windowSze=3
C0=20
C1=300
regularize_lambda=0.1
sigma=0.5
delta=0.85
showHazeTransmissionMap=True


'''
row, col = 1, 2
fig, axs = plt.subplots(row, col, figsize=(15, 10))
#fig.tight_layout()
axs[0].imshow(cv2.cvtColor(HazeImg, cv2.COLOR_BGR2RGB))
axs[0].set_title('original')
axs[0].set_axis_off()
axs[1].imshow(cv2.cvtColor(HazeCorrectedImg , cv2.COLOR_BGR2RGB))
axs[1].set_title('Denoised')
axs[1].set_axis_off()
plt.show()
'''



