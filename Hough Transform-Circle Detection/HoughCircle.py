import cv2
import numpy as np
import math
from numpy import unravel_index
img=cv2.imread("hough.jpg",0)

r=img.shape[0]
c=img.shape[1]

# Edge Detection
pad_0=[[0]*(c+2) for i in range(r+2)]

for i in range(1,r-1):
    for j in range(1,c-1):
        pad_0[i][j]=img[i-1][j-1]        
p=np.asarray(pad_0)
cv2.imwrite('imagePad.png',p)


Gx = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
Gy = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
Sx=list(map(list, zip(*Gx)))
Sy=list(map(list, zip(*Gy)))

slistX=[[0] * c for i in range(r)]
slistY=[[0] * c for i in range(r)]
    
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
cv2.imwrite('BonuXsob.png',s_x)
s_y=np.asarray(slistY)
cv2.imwrite('BonusYsob.png',s_y)


finalI=np.sqrt(np.square(s_x)+np.square(s_y))
cv2.imwrite('BonusXY.jpg',finalI)
normX=[[0] * c for i in range(r)]
normY=[[0] * c for i in range(r)]
F=np.zeros((r,c))

for i in range(0,r):
    for j in range(0,c):
        if(s_x[i][j]>180):
            normX[i][j]=255
        if(s_y[i][j]>180):
            normY[i][j]=255
        
nx=np.asarray(normX)
ny=np.asarray(normY)
cv2.imwrite('BonusXF.png',nx)
cv2.imwrite('BonusYF.png',ny)

for i in range(0,r):
    for j in range(0,c):
        if(ny[i][j]>180):
            F[i][j]=255
            
f=nx+ny
cv2.imwrite('BB.jpg',f)
cv2.imwrite('Bysob.jpg',F)
R=22
r=img.shape[0]
c=img.shape[1]
def find_rho(a1,b1,t):
    
    a = int((b1 - R * math.cos(math.radians(t))))
    b = int((a1 - R * math.sin(math.radians(t))))
    return a,b
def Accumulator(n):
    AccMat=np.zeros((2*r,2*c))
    count=0
    for x in range(0,r):
        for y in range(0,c):
            if n[x][y] == 255 :
                count=count+1
                for t in range(0,360):
                    a,b = find_rho(x,y,t)
                    AccMat[a][b]=AccMat[a][b]+1
    cv2.imwrite("BonusAcc.jpg",AccMat)   
    return AccMat
def plot(I,v,na):
    fim=cv2.imread("hough.jpg")   
    for z in range(v.shape[0]):
        rA,rB=v[z,0],v[z,1]
        #x=rA - R*math.cos
        imgc = cv2.circle(fim,(rA,rB),R,(0,255,0))
        cv2.imwrite("Bonusfinalx.jpg",imgc)

def max_indices(arr, k):
    assert k <= arr.size, 'k should be smaller or equal to the array size'
    arr_ = arr.astype(float)  # make a copy of arr
    max_idxs = []
    for _ in range(k):
        max_element = np.max(arr_)
        if np.isinf(max_element):
            break
        else:
            idx = np.where(arr_ == max_element)
        max_idxs.append(idx)
        arr_[idx] = -np.inf
    return max_idxs
def split(edge_points):
    k = []
    for i in range(edge_points.shape[0]):
        n = edge_points[i][0].shape[0]
        if n>1:
            for j in range(n):
                k.append([edge_points[i][0][j], edge_points[i][1][j]])
        else:
            k.append([edge_points[i][0][0], edge_points[i][1][0]])
    
    return np.asarray(k)

acx=Accumulator(f)
vArrx=np.asarray(max_indices(acx,45))
pt=split(vArrx)
plot(img,pt,"Bonusfinalx.jpg")

