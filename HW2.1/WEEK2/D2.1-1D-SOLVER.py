
import 	numpy 			as	np
import 	matplotlib.pyplot 	as 	plt

import 	scipy.optimize 
import 	torch 


#--------------------------------------------------------
##GENERATE DATA
#--------------------------------------------------------

def f(x): #parent function
	# out=np.sin(x)
	out=x*x
	return out
	
[A, xo, w, S]=[1.,0.,1.,-100] #PARAMETERS
def fp(x): #parameterized function
	return A*f((x-xo)/w)+S

#DEFINE DATA 
N=100; xmin=0; xmax=30
x = np.linspace(xmin,xmax,N)
y = fp(x)  #PRISTINE DATA

#INITIALIZE FIGURE 
plt.figure() 
FS=18   #FONT SIZE
plt.xlabel('x', fontsize=FS)
plt.ylabel('f(x)', fontsize=FS)


#--------------------------------------------------------
#APPLY SECANT METHOD TO FIND NEAREST ROOT 
#--------------------------------------------------------
dx=0.0001 		 						#STEP SIZE FOR FINITE DIFFERENCE
xi=2 #np.random.normal(0, 0.5,1)[0]     #INITIAL GUESS
#print(xi); exit()
t=0 	 
tmax=5 	 								#MAX NUMBER OF STEPS

while(t<=tmax):
	t=t+1
	yi=fp(xi)
	df_dx=(yi-fp(xi-dx))/(dx)
	xip1=xi-yi/df_dx #STEP
#	print(xi,yi,2*xi,df_dx,yi/df_dx)

	#PLOT CURRENT POINT
	plt.plot(x,y,'r-')
	plt.plot(x,0*y,'-')
	plt.plot(xip1*np.ones(10),np.linspace(0,fp(xip1),10),'-')
	plt.plot(xip1,fp(xip1),'bo')
	plt.plot(xi,yi,'ro')
	plt.plot(x,yi+df_dx*(x-xi),'b-')
	plt.pause(0.11)
	plt.show(); #exit()

	#UPDATE FOR NEXT ITERATION OF LOOP
	xi=xip1 


