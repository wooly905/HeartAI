import pydicom  # https://pydicom.github.io/pydicom/stable/index.html
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

print("cv2.__version__ : ", cv2.__version__)

# define constants
BlueColor = (0, 0, 255)
RedColor = (255, 0, 0)
GreenColor = (0, 255, 0)
dcmFolder = "c:\\temp2\\"
dcmFilePath =  os.path.join(dcmFolder, "01605286_112110_0592.DCM")  # get_testdata_files('MR_small.dcm')[0]
contourAreaThreshold = 2000

if (os.path.exists(dcmFilePath)):
    originalImageDataSet = pydicom.dcmread(dcmFilePath)

def DisplayImage(image1, image2):
    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.show()

def CreateCurrentDirectory(newDirName):
    currentDir = os.getcwd()
    newPath = os.path.join(currentDir, newDirName)
    os.mkdir(newPath)
    
def FilterContours(contours):
    result = list()
    for contour in contours:
        if (cv2.contourArea(contour) > contourAreaThreshold):
            result.append(contour)
    return result

def InvertImage(image):
    return (255-img)

def ConvertToGray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
def FilteringInternal(grayImage, blurSize, firstLowerValue, firstHighValue, secondLowerValue, secondHighValue):
    grayImageEqu = cv2.equalizeHist(grayImage)
    copyGrayImage = grayImage.copy()
    copyGrayImage = cv2.medianBlur(copyGrayImage, blurSize)
    copyGrayImage[grayImageEqu < firstLowerValue] = 0
    copyGrayImage[grayImageEqu >= firstHighValue] = 255
    grayImageEqu2 = cv2.equalizeHist(copyGrayImage)
    grayImageEqu2 = cv2.medianBlur(grayImageEqu2, blurSize)
    copyGrayImage[grayImageEqu2 < secondLowerValue] = 0
    copyGrayImage[grayImageEqu2 >= secondHighValue] = 255
    return copyGrayImage 

def FilteringMuscleView(image):
    grayImage = ConvertToGray(image)
    return FilteringInternal(grayImage, 7, 150, 200, 150, 200)
   
def FilteringChamberView(image):
    image = InvertImage(image)
    grayImage = ConvertToGray(image)
    return FilteringInternal(grayImage, 7, 50, 150, 20, 250)

def CropImage(image):
    x=0
    y=60  # This is hardcode for special DCIM files
    h=700
    w=800
    return image[y: y + h, x: x + w]

def FindCameraViewEdge(image):
    kernel = np.ones((3,3), np.uint8) 
    grayImage = ConvertToGray(image)
    grayImage = cv2.erode(grayImage, kernel, iterations=1)
    grayImage = cv2.dilate(grayImage, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(grayImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    edges = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 10000 :
            edges.append(cnt)
    return edges

def FillBlackOutsideHeartRange(contours,image):
    stencil = np.zeros(image.shape).astype(image.dtype)
    color = [255, 255, 255]
    cv2.fillPoly(stencil, contours, color)
    return cv2.bitwise_and(image, stencil)

# pixel_array needs Pillow package
imagePixels = originalImageDataSet.pixel_array
imageCount = imagePixels.shape[0]
index = 0

img = cv2.merge((imagePixels[index, :, :, 0], imagePixels[index, :, :, 1], imagePixels[index, :, :, 2]))
img = CropImage(img)
contourImage = img.copy()
edges = FindCameraViewEdge(img)
heartViewImage = FillBlackOutsideHeartRange(edges, img)

cv2.drawContours(contourImage, edges, -1, GreenColor, 2)
DisplayImage(heartViewImage, contourImage)
     
# while index < imageCount:
#     # merge BGR
#     img = cv2.merge((imagePixels[index, :, :, 0], imagePixels[index, :, :, 1], imagePixels[index, :, :, 2]))
#     img = CropImage(img)
#     contourImage = img.copy()
#     filteringChamberImage = FilteringChamberView(img)
#     contoursChamber, hierChanmber = cv2.findContours(filteringChamberImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
#     filteringMuscleImage = FilteringMuscleView(img)
#     contourMuscle, hierMuscle = cv2.findContours(filteringMuscleImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #cv2.RETR_EXTERNAL
#     qualifiedChamberContours = FilterContours(contoursChamber)
#     qualifiedMuscleContours = FilterContours(contourMuscle)
#     # draw contours
#     cv2.drawContours(contourImage, qualifiedChamberContours, -1, RedColor, 2)
#     cv2.drawContours(contourImage, qualifiedMuscleContours, -1, BlueColor, 2)
#     combinedImage = np.concatenate((img, contourImage), axis=1)
#     cv2.imwrite(os.path.join(dcmFolder, str(index) + ".png"), combinedImage)
#     print(str(index) + ".png is created.")
#     index = index + 1