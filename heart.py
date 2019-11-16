import pydicom  # https://pydicom.github.io/pydicom/stable/index.html
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

print("cv2.__version__ : ", cv2.__version__)

# define constants
RedColor = (0, 0, 255)
dcmFolder = "c:\\temp2\\"
dcmFilePath =  os.path.join(dcmFolder, "01499407_094944_0272.DCM")  # get_testdata_files('MR_small.dcm')[0]
contourAreaThreshold = 2000

if (os.path.exists(dcmFilePath)):
    originalImageDataSet = pydicom.dcmread(dcmFilePath)

def displayImage(image1, image2):
    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.show()

def createCurrentDirectory(newDirName):
    currentDir = os.getcwd()
    newPath = os.path.join(currentDir, newDirName)
    os.mkdir(newPath)
    
def filterContours(contours):
    result = list()
    for contour in contours:
        if (cv2.contourArea(contour) > contourAreaThreshold):
            result.append(contour)
    return result
    
# pixel_array needs Pillow package
imagePixels = originalImageDataSet.pixel_array
imageCount = imagePixels.shape[0]
index = 0

while index < 1: #imageCount:
    # merge BGR
    img = cv2.merge((imagePixels[index, :, :, 0], imagePixels[index, :, :, 1], imagePixels[index, :, :, 2]))
    #displayImage(img, img)
    contourImage = img.copy()
    invertedImg = (255 - img)
    #displayImage(invertedImg, invertedImg)
    # convert to gray
    grayImage = cv2.cvtColor(invertedImg, cv2.COLOR_BGR2GRAY)
    #displayImage(grayImage, grayImage)
    # equalizeHist
    grayImageEqu = cv2.equalizeHist(grayImage)
    # filtering
    copyGrayImage = grayImage.copy()
    copyGrayImage = cv2.medianBlur(copyGrayImage, 7)
    copyGrayImage[grayImageEqu < 50] = 0
    copyGrayImage[grayImageEqu >= 150] = 255
    #displayImage(grayImageEqu, copyGrayImage)
    # equalize histogram 
    grayImageEqu2 = cv2.equalizeHist(copyGrayImage)
    # filtering (blur and ...)
    grayImageEqu2 = cv2.medianBlur(grayImageEqu2, 7)
    copyGrayImage[grayImageEqu2 < 20] = 0
    copyGrayImage[grayImageEqu2 >= 250] = 255
    #displayImage(grayImageEqu2, copyGrayImage)
    contours, hier = cv2.findContours(copyGrayImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   #cv2.RETR_EXTERNAL
    print("length of all contours = " + str(len(contours)))
    qualifiedContours = filterContours(contours)
    print("length of qualified contours = " + str(len(qualifiedContours)))
    # draw contours
    cv2.drawContours(contourImage, qualifiedContours, -1, RedColor, 2)
    #displayImage(img, medianGrayImage)
    #combinedImage = np.concatenate((img, contourImage), axis=1)
    displayImage(img, contourImage)
    cv2.imwrite(str(index) + ".png", contourImage)
    index = index + 1