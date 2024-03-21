import numpy as np
import time
import matplotlib.pyplot as plt
import cupy as cp

#生成z坐标 x0,y0 为起始点， nx,ny为点数， delta为点距
def genZ(x0, y0, nx, ny, delta):
    real, img = cp.indices([nx,ny])*delta
    real += x0
    img += y0
    return real.T+img.T*1j

#获取Julia集，n为迭代次数，m为判定发散点，大于1即可
def getJulia(z,c,n,m=2):
    t = time.time()
    out = cp.abs(z)
    for i in range(n):
        absz = cp.abs(z)
        z[absz>m]=0     #对开始发散的点置零
        c[absz>m]=0
        out[absz>m]=i   #记录发散点的发散速度
        z = z*z + c
    print("time:",time.time()-t)
    return out

z1 = genZ(-3.0,-2.0,4000,4000,0.001)
mBrot = getJulia(z1,z1,50)
# scale
w = 24
h = 24
fig = plt.figure(figsize=(w/2.54,h/2.54))
ax = plt.axes()
Margin = [0.01, 0.99, 0.01, 0.99]
# plt.subplots_adjust(left=Margin[0], right=Margin[1], bottom=Margin[2],top=Margin[3])
im = plt.imshow(mBrot.get(), cmap=plt.cm.jet)
# fig.colorbar(im)
# plt.imshow([[0,1,2,3],[7,9,1289,20],[31,50,40,8],[35,42,45,49]], cmap = 'jet')
# plt.savefig("figure/Mandelbrot.pdf", dpi=1200)
# plt.savefig("figure/Mandelbrot.png", dpi=1200)
plt.show()
