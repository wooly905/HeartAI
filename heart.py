import pydicom  # https://pydicom.github.io/pydicom/stable/index.html
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

print("cv2.__version__ : ", cv2.__version__)

filename = "01499407_094944_0272.DCM"  # get_testdata_files('MR_small.dcm')[0]
ds = pydicom.dcmread(filename)

# def imageStruct(ds):
#     # get the pixel information into a numpy array
#     data = ds.pixel_array
#     print('The dataset has {} {}x{} images'.format(data.shape[0], data.shape[1], data.shape[2]))
#     #for i in range(0, data.shape[0]):
#     #    plt.imshow(data[i])
#     plt.subplot(2, 2, 1)
#     # plt.imshow(data[0, :, :, 0])
#     # plt.title('channel 0 (R)')
#     img = cv2.merge( (data[0, :, :, 0], data[0, :, :, 1], data[0, :, :, 2]) )
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     plt.imshow(gray_img)
#     plt.title('test')
#     # plt.subplot(2, 2, 2)
#     # plt.imshow(data[0, :, :, 1])
#     # plt.title('channel 1 (G)')
#     # plt.subplot(2, 2, 3)
#     # plt.imshow(data[0, :, :, 2])
#     # plt.title('channel 2 (B)')
#     plt.subplot(2, 2, 4)
#     plt.imshow(data[0])
#     plt.title('channel all')
#     plt.show()

#imageStruct(ds)

def displayImage(image1, image2):
    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.show()

def createCurrentDirectory(newDirName):
    currentDir = os.getcwd()
    os.mkdir(currentDir + '/' + newDirName)
    
print('Patient id : ' + ds.PatientID)
# create a new folder
#createCurrentDirectory(ds.PatientID)

# pixel_array needs Pillow package
data = ds.pixel_array
imageCount = data.shape[0]
index = 0

# while index < 1: #imageCount:
#     # merge BGR
#     img = cv2.merge((data[index, :, :, 0], data[index, :, :, 1], data[index, :, :, 2]))
#     contourImage = img.copy()
#     # convert to gray
#     grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # equalizeHist
#     grayImageEqu = cv2.equalizeHist(grayImage)
#     # filtering
#     copyGrayImage = grayImage.copy()
#     copyGrayImage = cv2.medianBlur(copyGrayImage, 7)
#     copyGrayImage[grayImageEqu < 150] = 0
#     copyGrayImage[grayImageEqu >= 200] = 255
#     grayImageEqu2 = cv2.equalizeHist(copyGrayImage)
#     copyGrayImage[grayImageEqu2 < 150] = 0
#     copyGrayImage[grayImageEqu2 >= 200] = 255
#     medianGrayImage = cv2.medianBlur(copyGrayImage, 7)
#     #displayImage(img, medianGrayImage)
#     contours, hier = cv2.findContours(medianGrayImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   #cv2.RETR_EXTERNAL
#     # draw contours
#     cv2.drawContours(medianGrayImage, contours, -1, (0, 0, 255), 2)
#     #displayImage(img, medianGrayImage)
#     cv2.imwrite(str(index) + ".png", medianGrayImage)
#     index = index + 1

while index < 1: #imageCount:
    # merge BGR
    img = cv2.merge((data[index, :, :, 0], data[index, :, :, 1], data[index, :, :, 2]))
    displayImage(img, img)
    contourImage = img.copy()
    invertedImg = (255 - img)
    displayImage(invertedImg, invertedImg)
    # convert to gray
    grayImage = cv2.cvtColor(invertedImg, cv2.COLOR_BGR2GRAY)
    displayImage(grayImage, grayImage)
    # equalizeHist
    grayImageEqu = cv2.equalizeHist(grayImage)
    # filtering
    copyGrayImage = grayImage.copy()
    copyGrayImage = cv2.medianBlur(copyGrayImage, 7)
    copyGrayImage[grayImageEqu < 50] = 0
    copyGrayImage[grayImageEqu >= 150] = 255
    displayImage(grayImageEqu, copyGrayImage)
    # equalize histogram 
    grayImageEqu2 = cv2.equalizeHist(copyGrayImage)
    # filtering (blur and ...)
    grayImageEqu2 = cv2.medianBlur(grayImageEqu2, 7)
    copyGrayImage[grayImageEqu2 < 20] = 0
    copyGrayImage[grayImageEqu2 >= 250] = 255
    displayImage(grayImageEqu2, copyGrayImage)
    contours, hier = cv2.findContours(copyGrayImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   #cv2.RETR_EXTERNAL
    # draw contours
    cv2.drawContours(contourImage, contours, -1, (0, 0, 255), 2)
    #displayImage(img, medianGrayImage)
    #combinedImage = np.concatenate((img, contourImage), axis=1)
    cv2.imwrite(str(index) + ".png", contourImage)
    index = index + 1