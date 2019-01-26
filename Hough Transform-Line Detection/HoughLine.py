import cv2
import numpy as np
import math
from numpy import unravel_index
img=cv2.imread("hough.jpg",0)
r=img.shape[0]
c=img.shape[1]
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
cv2.imwrite('imageXsob.png',s_x)
s_y=np.asarray(slistY)
cv2.imwrite('imageYsob.png',s_y)
normX=[[0] * c for i in range(r)]
normY=[[0] * c for i in range(r)]

for i in range(0,r):
    for j in range(0,c):
        if(s_x[i][j]>200):
            normX[i][j]=255
        if(s_y[i][j]>100):
            normY[i][j]=255
        

nx=np.asarray(normX)
ny=np.asarray(normY)
cv2.imwrite('imageXF.png',nx)
cv2.imwrite('imageYF.png',ny)
r=img.shape[0]
c=img.shape[1]
diag=int((np.sqrt((img.shape[0]**2)+(img.shape[1]**2))))
rv=2*diag-1
def find_rho(x,y,t):
    return int((x * math.cos(math.radians(t)) + y * math.sin(math.radians(t))+diag))
#Accumulator
def Accumulator(n,n1):
    #AccMat=[[0]*rv for i in range(0,180)]
    AccMat=np.zeros((180,rv))
    count=0
    for x in range(0,r):
        for y in range(0,c):
            if n[x][y] == 255 :
                count=count+1
                for t in range(-90,90):
                    #print(t)
                    #rho=int(round(y * math.cos(math.radians(t)) + x * math.sin(math.radians(t))+diag))
                    rho = find_rho(x,y,t)
                    #print(rho)
                    AccMat[t+90][rho]=AccMat[t+90][rho]+1

    #tmax,rmax=unravel_index(int(AccMat.max()),AccMat.shape)
    cv2.imwrite(n1,AccMat)
    
    return AccMat

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

def plot(n,valArr,na,naf):
    fim=cv2.imread("hough.jpg")
    wI=np.zeros((r,c))
    Alist=[]
    for x in range(0,r):
        for y in range(0,c):
            if n[x][y] == 255 :
                for z in range(valArr.shape[0]):
                    rhoR=find_rho(x,y,valArr[z,0]-90)
                    #rhoR = int(round(y * math.cos(math.radians(tmax90)) + x * math.sin(math.radians(tmax90))+diag))
                    #print(rhoR)
                    if rhoR == valArr[z,1] :
                        #print(rmax)
                        Alist.append([x,y])
                        wI[x][y]=255
                        fim[x][y]=(0,255,0)
                        #print(x,y)

    cv2.imwrite(na,wI)
    cv2.imwrite(naf,fim)
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
 
#X image
acx=Accumulator(nx,"AccRed.jpg")
vArrx=np.asarray(max_indices(acx,50))
pt=split(vArrx)
plot(nx,pt,"plotx.jpg","red_lines.jpg")
#Yimage
acy=Accumulator(ny,"AccBlue.jpg")
vArry=np.asarray(max_indices(acy,30))
pt=split(vArry)
plot(ny,pt,"ploty.jpg","blue_lines.jpg")

