# import numpy as np
# import cv2

# # Load an color image in grayscale
# img = cv2.imread('/Users/iwanahiroki/JFWM2019/ぱりぴ_180701_0126.png',0)
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import numpy as np
# import cv2
# from matplotlib import pyplot as plt

# img = cv2.imread('/Users/iwanahiroki/JFWM2019/ぱりぴ_180701_0126.png',0)
# plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('/Users/iwanahiroki/JFWM2019/ぱりぴ_180701_0126.png')
print(img.shape)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print(gray.shape)
cv2.imshow('color image',img)
cv2.imshow('gray image',gray)


plt.imshow(img)

cvimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(cvimg)

kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(cvimg,-1,kernel)
dst2 = cv2.GaussianBlur(cvimg,(5,5),0)
plt.figure(figsize=(16,8))
plt.subplot(131),plt.imshow(cvimg),plt.title('Original')
plt.xticks([]),plt.yticks([])
plt.subplot(132),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]),plt.yticks([])
plt.subplot(133),plt.imshow(dst2),plt.title('Averaging2')
plt.xticks([]),plt.yticks([])
plt.show()

laplacian = cv2.Laplacian(gray,cv2.CV_64F)
sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)
plt.figure(figsize=(16,8))
plt.subplot(1,4,1),plt.imshow(cvimg,    cmap = 'gray');plt.title('Original'),plt.xticks([]),plt.yticks([])
plt.subplot(1,4,2),plt.imshow(laplacian,cmap = 'gray');plt.title('Laplacian'),plt.xticks([]),plt.yticks([])
plt.subplot(1,4,3),plt.imshow(sobelx,   cmap = 'gray');plt.title('Sobel X'),plt.xticks([]),plt.yticks([])
plt.subplot(1,4,4),plt.imshow(sobely,   cmap = 'gray');plt.title('Sobel Y'),plt.xticks([]),plt.yticks([])

plt.show()
