

##-------------------------------------------
## VECTOR FIELD AND GRADIENT 
##-------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

FUNC=2
FS=18	#FONT SIZE

#DEFINE FUNCTION 
def f(x, y):
	if(FUNC==1): out=x**2+y**2
	if(FUNC==2): out=x*np.exp(-(x**2+y**2))
	return out

#DEFINE PARTIAL DERIVATIVES 
def fx(x,y):
	if(FUNC==1): out=2*x 
	if(FUNC==2): out=(1-2*x*x)*np.exp(-(x**2+y**2))
	return out

def fy(x,y):
	if(FUNC==1): out=2*y 
	if(FUNC==2): out=(-2*x*y)*np.exp(-(x**2+y**2))
	return  out


#MESH-1 (SMALLER)
xmin=-2; xmax=2; ymin=xmin; ymax=xmax
x,y = np.meshgrid(np.linspace(xmin,xmax,20),np.linspace(ymin,ymax,20))

#MESH-2 (DENSER)
X, Y = np.meshgrid(np.linspace(xmin, xmax, 40), np.linspace(ymin, ymax, 40))


#PLOT GRADIENT FIELD  
plt.quiver(x,y, fx(x,y), fy(x,y))
## plt.show()

#CONTOUR PLOT 
plt.contour(X, Y, f(X, Y), 20, cmap='RdGy'); 
plt.show(); 

#SURFACE PLOT 
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_xlabel('x', fontsize=FS); ax.set_ylabel('y', fontsize=FS); ax.set_zlabel('z', fontsize=FS)
surf=ax.plot_surface(X, Y, f(X, Y), cmap='RdGy') 
plt.show(); 


#GRADIENT-1 PLOTTING
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_xlabel('x', fontsize=FS); ax.set_ylabel('y', fontsize=FS); ax.set_zlabel('fx', fontsize=FS)
surf=ax.plot_surface(X, Y, fx(X, Y), cmap='RdGy') 
plt.show(); 

#GRADIENT-2 PLOTTING
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_xlabel('x', fontsize=FS); ax.set_ylabel('y', fontsize=FS); ax.set_zlabel('fy', fontsize=FS)
surf=ax.plot_surface(X, Y, fy(X, Y), cmap='RdGy') 
plt.show(); 


# #GRADIENT COMBINED PLOTTING
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.set_xlabel('x', fontsize=FS); ax.set_ylabel('y', fontsize=FS); ax.set_zlabel('|fy-fz|', fontsize=FS)
# # surf=ax.plot_surface(X, Y, 0*fx(X, Y)) 
# # surf=ax.plot_surface(X, Y, fx(X, Y)) 
# # surf=ax.plot_surface(X, Y, fy(X, Y), cmap='RdGy') 
# surf=ax.plot_surface(X, Y, ((fx(X, Y)-fy(X, Y))**2.0)**0.5, cmap='RdGy') 
# plt.show(); 

#SLICES
X0=1; Y0=1
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_xlabel('x', fontsize=FS); ax.set_ylabel('y', fontsize=FS); ax.set_zlabel('z', fontsize=FS)
surf=ax.plot_surface(X, Y, f(X, Y)) 
xtmp=np.linspace(xmin, xmax, 40)
LW=5
#ax.plot(xtmp,xtmp,f(xtmp, xtmp),'r-',linewidth=LW)
if(FUNC==2):
	ax.plot(0.0*xtmp+0.7,xtmp,f(0.0*xtmp+0.7, xtmp),'r-',linewidth=LW)
	ax.plot(0.0*xtmp-0.7,xtmp,f(0.0*xtmp-0.7, xtmp),'r-',linewidth=LW)
ax.plot(xtmp,0*xtmp,f(xtmp, 0*xtmp),'r-',linewidth=LW)
plt.show(); 

#TANGENT PLANE AT X0 Y0
X0=1; Y0=1
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_xlabel('x', fontsize=FS); ax.set_ylabel('y', fontsize=FS); ax.set_zlabel('z', fontsize=FS)
surf=ax.plot_surface(X, Y, f(X, Y), cmap='RdGy') 
surf=ax.plot_surface(X, Y, f(X0,Y0)+fx(X0,Y0)*(X-X0)+fy(X0,Y0)*(Y-Y0), cmap='RdGy') 
plt.show(); 

#exit()

##-------------------------------------------
## 2D SURFACE PLOT WITH MATPLOTLIB
##-------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
#-------------------------------------------
#VARIOUS MULTI-VARIABLE PLOTTING EXAMPLES
	#2021-08-03  JFH
#-------------------------------------------

#3D 
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def fun(x, y):
	#print(x.shape)
	#return np.sin(x)*np.sin(y)
	return np.exp(-((x-1)/0.75)**2)*np.exp(-(y/1.5)**2)
	#TODO: ADD MULTIVARIABLE GAUSSIAN WITH NON IDENTITY COVARIANCE MATRIX

#DEFINE FIGURE
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')

#DEFINE MESH FOR PLOTING
# np.arange --> Return evenly spaced values within a given interval.
xbound=4; dx=0.2; x = np.arange(-xbound,xbound, dx)
ybound=4; dy=0.2; y = np.arange(-ybound,ybound, dy)

#MERGE ARRAYS INTO 2D MESH
#np.meshgrid Return coordinate matrices from coordinate vectors.
X, Y = np.meshgrid(x, y)

#TO BETTER UNDERSTAND MESH, SET dx=dy=2 AND UNCOMMENT
# print(x.shape,y.shape,X.shape,Y.shape)
# print(x,y)
# print(X)
# print(Y)

#CONVERT MESH MATRIX TO VECTOR AND EVALUTE FUNCTION ON MESH COMPONENT WISE
zs = np.array(fun(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape) #RESHAPE TO MATCH ORIGINAL MESH

#3D PLOTS 
#ax.scatter(X, Y, Z+0.001, marker='o') # RAW DATA
# ax.contour3D(X, Y, Z, 50, cmap='binary') #CONTOUR

zshift=1 #shift for visulization

surf=ax.plot_surface(X, Y, Z+zshift, cmap=cm.coolwarm) #INTERPOLATED ONTO SMOOTH SURFACE AND APPLY COLOR MAP  
num_contour=5

#CONTOUR PROJECTIONS ON AXIS 
cset = ax.contourf(X, Y, Z+zshift, num_contour, zdir='z', offset=np.min(Z), cmap=cm.coolwarm)
cset = ax.contourf(X, Y, Z+zshift, zdir='x', offset=-xbound, cmap=cm.coolwarm)
cset = ax.contourf(X, Y, Z+zshift, zdir='y', offset=xbound, cmap=cm.coolwarm)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

#AXIS RANGE
ax.set_zlim(0,np.max(Z)+1.5*zshift)


FS=18	#FONT SIZE
plt.xticks(fontsize=FS); plt.yticks(fontsize=FS);  

ax.set_xlabel('X', fontsize=FS)
ax.set_ylabel('Y', fontsize=FS)
ax.set_zlabel('Z', fontsize=FS)

plt.show()


##-------------------------------------------
##SIMPLE 3D PLOT MOVIE: 
##-------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n = 100
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot a helix along the x-axis
theta_max = 8 * np.pi
theta = np.linspace(0, theta_max, n)
x = theta
z =  x*np.sin(theta)
y =  x*np.cos(theta)
ax.plot(x, y, z, 'b', lw=2)

dt=0.01		#time to pause between frames in seconds 

for i in range(0,len(x)):
	#ax.scatter(x[i],y[i],z[i],'ro')
	plt.plot(x[i],y[i],z[i],'ro')
	plt.pause(dt)

plt.show()

