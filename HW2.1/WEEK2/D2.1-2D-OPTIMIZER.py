
import 	numpy 			as	np
import 	matplotlib.pyplot 	as 	plt

#PARAM
FUNC=1
FS=18	
xmin=-2; xmax=2; ymin=xmin; ymax=xmax

#DEFINE FUNCTION TO MINIMIZE
def f(x,y):
	if(FUNC==1): out=x**2+y**2
	if(FUNC==2): out=x*np.exp(-(x**2+y**2))
	return out
 
#DEFINE PARTIAL DERIVATIVES  
	#NOT ACTUALLY USED EXCEPT TO COMPARE WITH NUMBERICAL PREDICTIONS
def fx(x,y):
	if(FUNC==1): out=2*x 
	if(FUNC==2): out=(1-2*x*x)*np.exp(-(x**2+y**2))
	return out

def fy(x,y):
	if(FUNC==1): out=2*y 
	if(FUNC==2): out=(-2*x*y)*np.exp(-(x**2+y**2))
	return  out

#MESH 
X, Y = np.meshgrid(np.linspace(xmin, xmax, 40), np.linspace(ymin, ymax, 40))

#SURFACE PLOT 
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_xlabel('x', fontsize=FS); ax.set_ylabel('y', fontsize=FS); ax.set_zlabel('z', fontsize=FS)
surf=ax.plot_surface(X, Y, f(X, Y), cmap='RdGy') 

print("#--------GRADIENT DECENT--------")

#PARAM
dx=0.001							#STEP SIZE FOR FINITE DIFFERENCE
LR=0.1								#LEARNING RATE
t=0 	 
tmax=1000							#MAX NUMBER OF STEPS
tol=10**-7							#EXIT AFTER CHANGE IN F IS LESS THAN THIS 

#INITIAL GUESS
# xi=np.random.uniform(xmin,xmax,2) #[0]							
xi=np.array([xmin,xmin])
# xi=np.array([xmax,xmax])

print("INITAL GUESS: ",xi)
ax.plot(xi[0],xi[1],f(xi[0],xi[1]),'bo')	#PLOT INITIal POINT

while(t<=tmax):
	t=t+1

	#NUMERICALLY COMPUTE GRADIENT 
	FX=(f(xi[0],xi[1])-f(xi[0]-dx,xi[1]))/(dx) 	#HOLD Y CONSTANT
	FY=(f(xi[0],xi[1])-f(xi[0],xi[1]-dx))/(dx) 	#HOLD Y CONSTANT
	df_dx=np.array([FX,FY])
	xip1=xi-LR*df_dx #STEP 

	if(t%10==0):
		df=np.absolute(f(xip1[0],xip1[1])-f(xi[0],xi[1]))
		print(t,"	",xi[0],"	",xi[1],"	",f(xi[0],xi[1])) #,"	",df)
	# 	print("	NUMERICAL fx=",FX,"	EXACT=",fx(xi[0],xi[1]))
	# 	print("	NUMERICAL fy=",FY,"	EXACT=",fy(xi[0],xi[1]))
	# 	#print("	shape xi",xi.shape, "shape grad",df_dx.shape)
		if(df<tol):
			print("STOPPING CRITERION MET (STOPPING TRAINING)")
			break
	ax.plot(xi[0],xi[1],f(xi[0],xi[1]),'ro')	#PLOT POINT

# 	#UPDATE FOR NEXT ITERATION OF LOOP
	xi=xip1 

plt.show(); #exit()

