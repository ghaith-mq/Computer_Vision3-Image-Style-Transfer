#!/usr/bin/env python
# coding: utf-8

# In[1479]:


import cv2 as cv
import io
import matplotlib.pyplot as plt
from skimage.exposure import  equalize_hist
from skimage.morphology import dilation, erosion, area_closing, area_opening
import numpy as np
from skimage.transform import rescale, resize, downscale_local_mean
import joblib

import numpy as np
from numpy import logical_and as land
from numpy import logical_not as lnot
from skimage.feature import canny
from skimage.transform import rescale, ProjectiveTransform, warp
from skimage.morphology import dilation, disk


# In[1513]:


im=cv.imread('Desktop/train33.jpg')


# In[1514]:


implt= cv.cvtColor(im, cv.COLOR_BGR2RGB)
plt.imshow(implt)


# In[1515]:


gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
plt.imshow(gray,cmap='gray')
blur = cv.blur(gray,(5,5))
threshad = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, 
                                      cv.THRESH_BINARY, 199, 5) 
outer=cv.bitwise_not(threshad,threshad)
outer_erode= erosion(outer)
areaArray = []
count = 0
ids=[]
contours, _ = cv.findContours(outer_erode, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
for i, c in enumerate(contours):
    area = cv.contourArea(c)
    areaArray.append(area)
    if(area>gray.shape[1]*100): #3,33: #100000 #train3333: 500000 #*100
        ids.append(i)
        count+=1    


# In[1516]:


count


# In[1526]:


mi=1e12
res=0
ids=np.array(ids)
if(len(ids)>1):
    for i in ids:
        if areaArray[i]<mi:
            mi=areaArray[i]
#             old_res=res
            res=i
#             out= np.zeros((gray.shape[0],gray.shape[1]))
#             # for cnt_id in ids:
#             perimeter = cv.arcLength(contours[res], True)
#             epsilon = 0.04* perimeter
#             approx = cv.approxPolyDP(contours[res],epsilon,True)
#             app=approx[:,0][:]
#             if(len(app)!=4):
#                 res=old_res
else:
    res=ids[0]


# In[1527]:


ids


# In[1528]:


def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates (tr,tl,br,bl)
    return rect


# In[1529]:


def four_point_transform(image, points):
    rect = order_points(points)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))


    dst = np.array([[0, 0],
    [0, maxWidth - 1],
    [ maxHeight - 1,maxWidth - 1],
    [ maxHeight - 1,0]], dtype = "float32")

    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxHeight,maxWidth))
    return warped


# In[1530]:


# ids=np.array(ids)
# flag=1   #this flag is used to point that only one painting detected,
# if(len(ids)==1):  #it is unset if more than one painting were detected so it needs filering
#     flag=0
# sud_count=0
out= np.zeros((gray.shape[0],gray.shape[1]))
# for cnt_id in ids:
perimeter = cv.arcLength(contours[res], True)
epsilon = 0.04* perimeter #0.04
approx = cv.approxPolyDP(contours[res],epsilon,True)
app=approx[:,0][:]
perimeter = cv.arcLength(contours[res], True)
points=app
warp= four_point_transform(implt, points)
# if(abs(warp.shape[0]-warp.shape[1]) >400) & flag:    #param can be adjusted
#     continue
warped=warp.copy()
h=warped.shape[0]
w=warped.shape[1]
warped = cv.transpose(warped)
cv.fillPoly(out, pts=[points], color=(255,255,255))


# In[1531]:


plt.imshow(warped)


# In[1532]:


app


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[1423]:


# gray = cv.cvtColor(warped, cv.COLOR_RGB2GRAY)
# blur = cv.blur(gray,(5,5))
# threshad = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, 
#                                       cv.THRESH_BINARY, 199, 5) 
# outer=cv.bitwise_not(threshad,threshad)
# outer_erode= erosion(outer)
# outer_erode= erosion(outer_erode)
# plt.imshow(outer_erode)
# areaArray = []
# count = 0
# ids=[]
# cntr=[]
# contours, hierarchy = cv.findContours(outer_erode, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
# for i in range(len(contours)):
#     if (hierarchy[0][i][2] == -1) :
#         cntr.append(contours[i])
# for i, c in enumerate(cntr):
#     area = cv.contourArea(c)
#     areaArray.append(area)
#     if(area>26000):  #100000
#         ids.append(i)
#         count+=1    


# In[1007]:


count


# In[1008]:


for cnt_id in ids:
    perimeter = cv.arcLength(cntr[cnt_id], True)
    epsilon = 0.04* perimeter
    approx = cv.approxPolyDP(cntr[cnt_id],epsilon,True)
    app=approx[:,0][:]
    perimeter = cv.arcLength(cntr[cnt_id], True)
    points=app
    final= four_point_transform(warped, points)
    h=final.shape[0]
    w=final.shape[1]
    final = cv.transpose(final)


# In[1009]:


plt.imshow(final)


# In[ ]:





# In[ ]:




