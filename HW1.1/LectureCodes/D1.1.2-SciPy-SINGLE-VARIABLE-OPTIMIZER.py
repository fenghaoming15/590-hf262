# #--------------------------------
# #OPTIMZE A FUNCTION USING SCIPY
# #--------------------------------

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from   scipy.optimize import minimize


# ax.plot(xe, ye, '-', label='Ground-Truth')
# ax.plot(xe, model(xe, popt), 'r-', label="Model")



#FUNCTION TO OPTIMZE
def f(x):
	out=x**2.0
	# out=(x+10*np.sin(x))**2.0
	return out

#PLOT
#DEFINE X DATA FOR PLOTTING
N=1000; xmin=-20; xmax=20
X = np.linspace(xmin,xmax,N)

plt.figure() #INITIALIZE FIGURE 
FS=18   #FONT SIZE
plt.xlabel('x', fontsize=FS)
plt.ylabel('f(x)', fontsize=FS)
plt.plot(X,f(X),'-')

num_func_eval=0
def f1(x): 
	global num_func_eval
	out=f(x)
	num_func_eval+=1
	if(num_func_eval%10==0):
		print(num_func_eval,x,out)
	plt.plot(x,f(x),'ro')
	plt.pause(0.11)

	return out

#INITIAL GUESS 
xo=xmax #
#xo=np.random.uniform(xmin,xmax)
print("INITIAL GUESS: xo=",xo, " f(xo)=",f(xo))
res = minimize(f1, xo, method='Nelder-Mead', tol=1e-5)
popt=res.x
print("OPTIMAL PARAM:",popt)

plt.show()
