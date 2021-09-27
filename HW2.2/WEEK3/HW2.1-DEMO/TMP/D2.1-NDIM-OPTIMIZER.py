
import 	numpy 			as	np
import 	matplotlib.pyplot 	as 	plt

#PARAM
xmin=-50; xmax=50;  
NDIM=5
xi=np.random.uniform(xmin,xmax,NDIM) #INITIAL GUESS FOR OPTIMIZEER							
if(NDIM==2): xi=np.array([-2,-2])

#DEFINE FUNCTION TO MINIMIZE
def f(x): 	#NON BATCH IMPLEMENTATION
	if(len(x)!=2 and len(x)!=3 and len(x)!=5): print("ERROR (INPUT WRONG SIZE)"); exit()

	#R2 --> R1
	if(len(x)==2):  out=x[0]*np.exp(-((x[0]/1.)**2+(x[1]/1.)**2))

	#R3 --> R1
	if(len(x)==3):  out=(x[0]-1.0)**2+(x[1]-2.0)**2+(x[2]-3.0)**2-4
	
	#R5 --> R1
	if(len(x)==5): out=x[0]**2+x[1]**2+x[2]**2+x[3]**2+x[4]**2

	return out
 

print("#--------GRADIENT DECENT--------")

#PARAM
dx=0.001							#STEP SIZE FOR FINITE DIFFERENCE
LR=0.0001								#LEARNING RATE
t=0 	 							#INITIAL ITERATION COUNTER
tmax=100000							#MAX NUMBER OF ITERATION
tol=10**-10							#EXIT AFTER CHANGE IN F IS LESS THAN THIS 

print("INITAL GUESS: ",xi)

while(t<=tmax):
	t=t+1

	#NUMERICALLY COMPUTE GRADIENT 
	df_dx=np.zeros(NDIM)
	for i in range(0,NDIM):
		dX=np.zeros(NDIM);
		dX[i]=dx; 
		xm1=xi-dX; #print(xi,xm1,dX,dX.shape,xi.shape)
		df_dx[i]=(f(xi)-f(xm1))/dx
	#print(xi.shape,df_dx.shape)
	xip1=xi-LR*df_dx #STEP 

	if(t%10==0):
		df=np.mean(np.absolute(f(xip1)-f(xi)))
		print(t,"	",xi,"	","	",f(xi)) #,df) 

		if(df<tol):
			print("STOPPING CRITERION MET (STOPPING TRAINING)")
			break

	#UPDATE FOR NEXT ITERATION OF LOOP
	xi=xip1

exit()

