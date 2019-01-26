import cv2
import numpy as np
import time
import math
from itertools import repeat

#def SobelGx(p)
#	aGx=((-1*p)+(0*p)+(1*p)+(-2*p)+(0*p)+(2*p)+(-1*p)+(0*p)+(1*p))
#def SobelGy(p)
#	aGy=((1*p)+(-2*p)+(-1*p)+(0*p)+(0*p)+(0*p)+(1*p)+(2*p)+(1*p))
Gx = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
Gy = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]


#greyscale image
img = cv2.imread("task1.png", 0)
print(img.size)
#print(img)
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# time.sleep(2)
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite()

#FLipping the sobel operators
Sx=list(map(list, zip(*Gx)))
Sy=list(map(list, zip(*Gy)))
print(Sx,Sy)


#pad_0 = np.zeros(img.shape)
img_list=list()
#print(img.shape)
for i in img:
    img_list.append(i)

#print(type(img_list))
x=np.asarray(img_list)
cv2.imshow('image', x)
cv2.waitKey(0)
cv2.destroyAllWindows()

#padding

c=len(img_list[0])
r=len(img_list)
print(c,r)

#zrow = [[0] * i for i in range(list_c)]
pad_0=[[0]*(c+2) for i in range(r+2)]

for i in range(1,r-1):
    for j in range(1,c-1):
        pad_0[i][j]=img_list[i-1][j-1]
        
p=np.asarray(pad_0)
cv2.imwrite('im_padding.png',p)
print(len(pad_0))


#computing gradient
list_c=len(img_list[0])
list_r=len(img_list)
print(list_c,list_r)
grx=[[0] * list_c for i in range(list_r)]
gry=[[0] * list_c for i in range(list_r)]
for i in range(1,list_r-1):
    for j in range(1,list_c-1):
        
        gry[i][j]=img_list[i+1][j]-img_list[i-1][j]
        
        grx[i][j]=img_list[i][j+1]-img_list[i][j-1]
        
x_grad=np.asarray(grx)
y_grad=np.asarray(gry)
cv2.imwrite('im_Xgrad.png',x_grad)
cv2.imwrite('im_Ygrad.png',y_grad)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#sobel 
#Sx=[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] 
#Sy=[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

slistX=[[0] * list_c for i in range(list_r)]
new_slistX=[[0] * list_c for i in range(list_r)]
slistY=[[0] * list_c for i in range(list_r)]
new_slistY=[[0] * list_c for i in range(list_r)]
for x in range(1,r-1):
        for y in range(1,c-1):
            sumx=0
            sumy=0
            for k in range(0,3):
                for l in range(0,3):
                    sumx+=(pad_0[x+k-1][y+l-1]*Sx[k][l])
                    sumy+=(pad_0[x+k-1][y+l-1]*Sy[k][l])
            slistX[x-1][y-1]=sumx
            slistY[x-1][y-1]=sumy


s_x=np.asarray(slistX)
cv2.imwrite('im_Xsob.png',s_x)
s_y=np.asarray(slistY)
cv2.imwrite('im_Ysob.png',s_x)

# cv2.imshow('image', s_x)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imshow('image', s_y)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#Normalisation
ma=list()
normX=[[0] * list_c for i in range(list_r)]
normY=[[0] * list_c for i in range(list_r)]
for m in slistX:
    for n in m:
        ma.append(n)
for m in slistY:
    for n in m:
        ma.append(n)
print(ma[0])
#ma=[j for sub in slistX for j in sub]    
    
y=0
l=0
for l in ma:
    if l<0:
        x=l*(-1)
    else:
        x=l
    if x>y:
        y=x
        
#print(max_no)

for i in range(0,list_r):
    for j in range(0,list_c):
        normX[i][j]=((slistX[i][j])*255)/y
        normY[i][j]=((slistY[i][j])*255)/y
        
f_x=np.asarray(normX)
f_y=np.asarray(normY)
cv2.imwrite('im_XEdge.png',f_x)
cv2.imwrite('im_YEdge.png',f_y)

