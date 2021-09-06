#--------------------------------
#SIMPLE UNIVARIABLE REGRESSION EXAMPLE
#--------------------------------

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from   scipy.optimize import curve_fit

#------------------------
#GENERATE TOY DATA
#------------------------

#GROUND TRUTH FUNCTION
def G(x):
	GT=1
	if(GT==1):
		out=400000.0*np.exp(-((x-0.0)/2.0)**2.0)
		out=out+150000.0*np.exp(-((x-15.0)/1.0)**2.0)
		out=out+100000
	if(GT==2):
		out=400000.0*np.exp(-((x+2.0)/4.0)**4.0)
		out=out+250000.0*np.exp(-((x-5.0)/1.0)**4.0)
		out=out+150000.0*np.exp(-((x-15.0)/1.0)**4.0)
		out=out+100000
	return out

#NOISY DATA
N=100; xmin=-10; xmax=10
x = np.linspace(xmin,xmax,N)
y = G(x)  #PRISTINE DATA
#noise=np.random.normal(loc=0.0, scale=0.05*(max(y)-min(y)),size=len(x))
noise=0.15*(max(y)-min(y))*np.random.uniform(-1,1,size=len(x))
yn = y+ noise

#GROUND TRUTH DATA
xe = np.linspace(xmin,2*xmax,int(2*N))
ye = G(xe)

#------------------------
#FIT MODEL
#------------------------

##FITTING MODEL
def model(x,p1,p2,p3,p4,p5,p6,p7):
    return p1*np.exp(-((x-p2)/p3)**2.0)+p4*np.exp(-((x-p5)/p6)**2.0)+p7
popt, pcov = curve_fit(model, x, yn) #,[0.1,0.1,0.1])

#------------------------
#PLOT
#------------------------

fig, ax = plt.subplots()
ax.plot(x, yn, 'o', label='Data')
ax.plot(xe, ye, '-', label='Ground-Truth')
ax.plot(xe, model(xe, *popt), 'r-', label="Model")

ax.legend()
FS=18   #FONT SIZE
plt.xlabel('Distance (miles)', fontsize=FS)
plt.ylabel('House Price ($) ', fontsize=FS)

plt.show()

