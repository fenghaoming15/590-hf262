
#-------------------------------------------
#VARIOUS UNIVARIABLE PLOTTING EXAMPLES
	#2021-08-03  JFH
#-------------------------------------------

##-------------------------------------------
##SIMPLE STATIC 1D PLOT 
##-------------------------------------------

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#DEFINE DATA
x = np.linspace(0, 10,200)
y  = np.sin(x)
y1 = y + np.random.normal(0, 0.5, len(x)) #ADD NOISE

#PLOT 
fig, ax = plt.subplots()
ax.plot(x,y,'-',x,y1,'o')
plt.show()
#exit()

##-------------------------------------------
##SIMPLE PLOT MOVIE: PLOTING FRAMES FROM LOOP 
##-------------------------------------------


import numpy as np
import matplotlib.pyplot as plt

#PARAMETERS
dt=0.01		#time to pause between frames in seconds 
x0=0.0; dx=0.1  #mesh parameters
Nframe=100 

plt.figure() #INITIALIZE FIGURE 
FS=18
plt.xlabel('Time (s)', fontsize=FS)
plt.ylabel('Amplitude (cm)', fontsize=FS)
for i in range(0,Nframe):
	x=x0+i*dx; y=np.sin(x)
	plt.plot(x,y,'bo')
	plt.pause(dt)
plt.show()
#exit()

##-------------------------------------------
##SUBPLOT PLOT MOVIE: PLOTING FRAMES FROM LOOP 
##-------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import time

#PARAMETERS
dt=0.005		#time to pause between frames in seconds 
NP=100 

x = np.linspace(0,10,NP)
y = np.exp(-0.1*x)*np.sin(5*x)

plt.ion()  #Enable interactive mode.
fig,ax = plt.subplots(3,1,figsize=(15,15))
ax[0].plot(x,y,'-')
ax[1].plot(x,y,'-',)

plt.show()

for i in range(0,len(x)):

	ax[0].clear()

	ax[0].plot(x,y,'-'); ax[0].plot(x[i],y[i],'bo')
	ax[1].plot(x[i],y[i],'ro')
	ax[2].plot(x[i],y[i],'ro')

	plt.draw()
	plt.pause(dt)
#exit()


##-------------------------------------------
## SMOOTHING+SAVING A PLOT TO PNG AND PDF FILES
##-------------------------------------------

#NOTE: PDF IS A "VECTOR IMAGES" 
#i.e will scale well and look good in either large or small windows without becoming grainy


import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d #,spline 
import scipy.signal
import warnings
from matplotlib.ticker import FormatStrFormatter

warnings.filterwarnings("ignore") #supress all warnings

#GENERATE DATA
N=1002	#number of data points 
x=np.linspace(-45,45.0, N); 
#ye=np.sin(x) #exact
ye=x*x*np.sin(x) #exact
y=ye+np.random.normal(0, 100, N)	#add noise

#DEFINE
f = plt.figure()
f, ax = plt.subplots()

#AXES
FS=18	#FONT SIZE
plt.xlabel('Time (ps)', fontsize=FS)
plt.ylabel('Amplitude (cm)', fontsize=FS)
plt.xticks([-40,-20,0,20,40],fontsize=FS)
plt.yticks([-2000,-1000,0,1000,2000],fontsize=FS)

#GENERATE SMOOTH LINE THROUHG NOISY DATA

#SMOTHING METHOD-1: savgol_filter
window=61 #window must me odd and requires adjustment depending on plot

#SMOOTHED DATA (SAME NUMBER OF POINTS AS y)
#https://riptutorial.com/scipy/example/15878/using-a-savitzky-golay-filter
ys = scipy.signal.savgol_filter(y, window, 4)  # window size , polynomial order

# #QUADRATICALLY INTERPOLATE THE savgol_filter DATA ONTO LESS DENSE MESH 
xs1=np.linspace(min(x), max(x), int(0.25*len(x)))
F=interp1d(x, ys, kind='quadratic');
ys1=F(xs1); 

#PLOT
plt.plot(x, y,'.', markersize=16,color='black',markerfacecolor='white',label="raw data") # ,color='black', markersize=8)
plt.plot(x,ye,'r-',linewidth=3,label="ground truth") 
# plt.plot(x,ys,'*',color='blue',linewidth=1.0,label="savgol smoothing") 
plt.plot(xs1,ys1,'*',color='blue',linewidth=1.0,label="savgol smoothing")  
ax.legend()

#PLOT RANGES
plt.xlim(min(x),0)

#CONTROL AXIS TICK FORMAT
ax.yaxis.set_major_formatter(FormatStrFormatter('%4.1f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

#SAVE IMAGES
f.savefig('plot-1.png', bbox_inches='tight')
f.savefig("plot-1.pdf", bbox_inches='tight')
exit()


# ###-------------------------------------------
# ###FUNC-ANIMATION MULTIPLE ANIMATIONS (TAKEN FROM ONLINE)
# ###-------------------------------------------

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # https://brushingupscience.com/2016/06/21/matplotlib-animations-the-easy-way/

# # Example animations using matplotlib's FuncAnimation
# # Ken Hughes. 18 June 2016.

# # For more detail, see
# # https://brushingupscience.wordpress.com/2016/06/21/matplotlib-animations-the-easy-way/

# # Examples include
# #    - line plot
# #    - pcolor plot
# #    - scatter plot
# #    - contour plot
# #    - quiver plot
# #    - plot with changing labels

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # Use matplotlib ggplot stylesheet if available
# try:
#    plt.style.use('ggplot')
# except:
#    pass

# # Set which type of animation will be plotted. One of:
# # line, pcolor, scatter, contour, quiver, labels
# animation_type = 'line'

# # ----------------------------------------------------------------------------
# # Create data to plot. F is 2D array. G is 3D array

# # Create a two-dimensional array of data: F(x, t)
# x = np.linspace(-3, 3, 91)
# t = np.linspace(0, 25, 30)
# X2, T2 = np.meshgrid(x, t)
# sinT2 = np.sin(2*np.pi*T2/T2.max())
# F = 0.9*sinT2*np.sinc(X2*(1 + sinT2))

# # Create three-dimensional array of data G(x, z, t)
# x = np.linspace(-3, 3, 91)
# t = np.linspace(0, 25, 30)
# y = np.linspace(-3, 3, 91)
# X3, Y3, T3 = np.meshgrid(x, y, t)
# sinT3 = np.sin(2*np.pi*T3 /
#               T3.max(axis=2)[..., np.newaxis])
# G = (X3**2 + Y3**2)*sinT3

# # ----------------------------------------------------------------------------
# # Set up the figure and axis
# fig, ax = plt.subplots(figsize=(4, 3))

# if animation_type not in ['line', 'scatter']:
#    ax.set_aspect('equal')


# # ----------------------------------------------------------------------------
# if animation_type == 'line':
#    ax.set(xlim=(-3, 3), ylim=(-1, 1))

#    line = ax.plot(x, F[0, :], color='k', lw=2)[0]

#    def animate(i):
#        line.set_ydata(F[i, :])

# # ----------------------------------------------------------------------------
# if animation_type == 'pcolor':
#    cax = ax.pcolormesh(x, y, G[:-1, :-1, 0], vmin=-1, vmax=1, cmap='Blues')
#    fig.colorbar(cax)

#    def animate(i):
#        cax.set_array(G[:-1, :-1, i].flatten())

# # ----------------------------------------------------------------------------
# if animation_type == 'scatter':
#    ax.set(xlim=(-3, 3), ylim=(-1, 1))
#    scat = ax.scatter(x[::3], F[0, ::3])

#    def animate(i):
#        # Must pass scat.set_offsets an N x 2 array
#        y_i = F[i, ::3]
#        scat.set_offsets(np.c_[x[::3], y_i])

# # ----------------------------------------------------------------------------
# if animation_type == 'contour':
#    # Keyword options used in every call to contour
#    contour_opts = {'levels': np.linspace(-9, 9, 10), 'cmap':'RdBu', 'lw': 2}
#    cax = ax.contour(x, y, G[..., 0], **contour_opts)

#    def animate(i):
#        ax.collections = []
#        ax.contour(x, y, G[..., i], **contour_opts)

# # ----------------------------------------------------------------------------
# if animation_type == 'quiver':
#    ax.set(xlim=(-4, 4), ylim=(-4, 4))

#    # Plot every 20th arrow
#    step = 15
#    x_q, y_q = x[::step], y[::step]

#    # Create U and V vectors to plot
#    U = G[::step, ::step, :-1].copy()
#    V = np.roll(U, shift=4, axis=2)

#    qax = ax.quiver(x_q, y_q, U[..., 0], V[..., 0], scale=100)

#    def animate(i):
#        qax.set_UVC(U[..., i], V[..., i])

# # ----------------------------------------------------------------------------
# if animation_type == 'labels':
#    ax.set(xlim=(-1, 1), ylim=(-1, 1))
#    string_to_type = 'abcdefghijklmnopqrstuvwxyz0123'
#    label = ax.text(0, 0, string_to_type[0],
#                    ha='center', va='center',
#                    fontsize=12)

#    def animate(i):
#        label.set_text(string_to_type[:i+1])
#        ax.set_ylabel('Time (s): ' + str(i/10))
#        ax.set_title('Frame ' + str(i))

# # ----------------------------------------------------------------------------
# # Save the animation
# anim = FuncAnimation(fig, animate, interval=100, frames=len(t)-1, repeat=True)
# fig.show()
# plt.show()
# # anim.save(animation_type + '.gif', writer='imagemagick')
# exit()


