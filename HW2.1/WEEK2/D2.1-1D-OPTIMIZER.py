
import 	numpy 			as	np
import 	matplotlib.pyplot 	as 	plt

#------------------------------------ 
##SETUP
#------------------------------------ 

def f(x): #parent function
	# out=np.sin(x)
	out=x*x
	return out
	
[A, xo, w, S]=[1.,10.,1.,-100] #PARAMETERS
def fp(x): #parameterized function
	return A*f((x-xo)/w)+S

#DEFINE DATA 
N=100; xmin=0; xmax=50
x = np.linspace(xmin,xmax,N)
y = fp(x)  #PRISTINE DATA

#INITIALIZE FIGURE 
plt.figure() 
FS=18   #FONT SIZE
plt.xlabel('x', fontsize=FS)
plt.ylabel('f(x)', fontsize=FS)
plt.plot(x,y,'b-')


#SECANT METHOD 
print("#--------SECANT METHOD--------")
dx=0.0001 		 					#STEP SIZE FOR FINITE DIFFERENCE
xi=100*xmax 						#INITIAL GUESS
#print(xi); exit()
t=0 	 
tmax=5								#MAX NUMBER OF STEPS


while(t<=tmax):
	t=t+1
	yi=fp(xi)

	#APPROXIMATE FIRST AND SECOND DERIVATE VIA FINITE DIFFERENCE
	df_dx=(yi-fp(xi-dx))/(dx)
	d2f_dx2=(fp(xi+dx)-2.0*yi+fp(xi-dx))/(dx*dx)

	xip1=xi-df_dx/d2f_dx2 #STEP (C)
	print(xi,yi,"step=",df_dx/d2f_dx2)

	#UPDATE FOR NEXT ITERATION OF LOOP
	xi=xip1 


print("#--------GRADIENT DECENT--------")
LR=0.1								#LEARNING RATE
t=0 	 
tmax=25								#MAX NUMBER OF STEPS
xi=xmax 							#INITIAL GUESS


while(t<=tmax):
	t=t+1
	yi=fp(xi)
	df_dx=(yi-fp(xi-dx))/(dx)
	xip1=xi-LR*df_dx #STEP (C)
	print(xi,yi,"step=",LR*df_dx )

	# #PLOT CURRENT POINT
	plt.plot(np.linspace(xi,xip1,10),yi*np.ones(10),'r-')
	plt.plot(xip1*np.ones(10),np.linspace(fp(xip1),yi,10),'r-')

	plt.plot(xi,yi,'ro')
	plt.pause(0.11)

	#UPDATE FOR NEXT ITERATION OF LOOP
	xi=xip1 


plt.show(); #exit()
# I HAVE WORKED THROUGH THIS EXAMPLE AND UNDERSTAND EVERYTHING THAT IT IS DOING

