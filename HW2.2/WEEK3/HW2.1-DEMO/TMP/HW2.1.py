import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
from   scipy.optimize import minimize

#USER PARAMETERS
IPLOT=True
INPUT_FILE='weight.json'
FILE_TYPE="json"
DATA_KEYS=['x','is_adult','y']
OPT_ALGO='BFGS'

#UNCOMMENT FOR VARIOUS MODEL CHOICES (ONE AT A TIME)
model_type="logistic"; NFIT=4; xcol=1; ycol=2;
# model_type="linear";   NFIT=2; xcol=1; ycol=2; 
# model_type="logistic";   NFIT=4; xcol=2; ycol=0;

#READ FILE
with open(INPUT_FILE) as f:
	my_input = json.load(f)  #read into dictionary

#CONVERT INPUT INTO ONE LARGE MATRIX (SIMILAR TO PANDAS DF)
X=[];
for key in my_input.keys():
	if(key in DATA_KEYS): X.append(my_input[key])

#MAKE ROWS=SAMPLE DIMENSION (TRANSPOSE)
X=np.transpose(np.array(X))

#SELECT COLUMNS FOR TRAINING 
x=X[:,xcol];  y=X[:,ycol]

#EXTRACT AGE<18
if(model_type=="linear"):
	y=y[x[:]<18]; x=x[x[:]<18]; 


#COMPUTE BEFORE PARTITION AND SAVE FOR LATER
XMEAN=np.mean(x); XSTD=np.std(x)
YMEAN=np.mean(y); YSTD=np.std(y)


#NORMALIZE
x=(x-XMEAN)/XSTD;  y=(y-YMEAN)/YSTD; 

#PARTITION
f_train=0.8; f_val=0.2
rand_indices = np.random.permutation(x.shape[0])
CUT1=int(f_train*x.shape[0]); 
train_idx,  val_idx = rand_indices[:CUT1], rand_indices[CUT1:]
xt=x[train_idx]; yt=y[train_idx]; xv=x[val_idx];   yv=y[val_idx]

#MODEL
def model(x,p):
	if(model_type=="linear"):   return  p[0]*x+p[1]  
	if(model_type=="logistic"): return  p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.01))))

#SAVE HISTORY FOR PLOTTING AT THE END
iteration=0; iterations=[]; loss_train=[];  loss_val=[]

#LOSS FUNCTION
def loss(p, xt2,yt2):
	#TRAINING LOSS
	yp=model(xt2,p) #model predictions for given parameterization p
	training_loss=(np.mean((yp-yt2)**2.0))  #MSE
	return training_loss


#INITIAL GUESS
po=np.random.uniform(0.1,1.,size=NFIT)

#TRAIN MODEL USING SCIPY MINIMIZ 
# res = minimize(loss, po, method=OPT_ALGO, tol=1e-15);  popt=res.x
# print("OPTIMAL PARAM:",popt)

PARADIGM='stocastic'
def optimizer(f,po):
	global iterations,loss_train,loss_val,iteration

	print("#--------GRADIENT DECENT--------")

	#PARAM
	dx=0.001							#STEP SIZE FOR FINITE DIFFERENCE
	LR=0.005								#LEARNING RATE
	iteration=0 	 							#INITIAL ITERATION COUNTER
	max_iter=50000							#MAX NUMBER OF ITERATION
	tol=10**-10							#EXIT AFTER CHANGE IN F IS LESS THAN THIS 
	xi=po
	NDIM=po.shape[0]
	# print(po.shape,NDIM,po.shape[0])
	# exit()
	print("INITAL GUESS: ",xi)

	while(iteration<=max_iter):

		if(PARADIGM=='batch'):
			if(iteration==0): xt2=xt; yt2=yt
		if(PARADIGM=='minibatch'):
			if(iteration==0):
				batch_size=int(xt.shape[0]/2)
				indices_1=np.random.choice(xt.shape[0], batch_size, replace=False)
				all_indices=[*range(0, xt.shape[0], 1)]
				indices_2=list(set(all_indices) - set(indices_1))

				xb1=xt[indices_1]; yb1=yt[indices_1]
				xb2=xt[indices_2]; yb2=yt[indices_2]
				xt2=xb1;  yt2=yb1;
			else:
				if(iteration%2==0):
					xt2=xb1;  yt2=yb1;
				else:
					xt2=xb2;  yt2=yb2;
		if(PARADIGM=='stocastic'):
			if(iteration==0):  
				index_2_use=0
			else:
				if(index_2_use<xt.shape[0]-1):
					index_2_use+=1
				else:
					index_2_use=0

			# print(index_2_use)

			xt2=xt[index_2_use]; yt2=yt[index_2_use]

			#exit()
				# exit()
# xt2=xt; yt2=yt


		#NUMERICALLY COMPUTE GRADIENT 
		df_dx=np.zeros(NDIM)
		for i in range(0,NDIM):
			dX=np.zeros(NDIM);
			dX[i]=dx; 
			xm1=xi-dX; #print(xi,xm1,dX,dX.shape,xi.shape)
			df_dx[i]=(f(xi,xt2,yt2)-f(xm1,xt2,yt2))/dx
		#print(xi.shape,df_dx.shape)
		xip1=xi-LR*df_dx #STEP 

		if(iteration%1==0):

			#TOTAL TRAING MSE
			yp=model(xt,xi) #model predictions for given parameterization p
			MSE_T=(np.mean((yp-yt)**2.0))  #MSE

			#VALIDATION MSE
			yp=model(xv,xi) #model predictions for given parameterization p
			MSE_V=(np.mean((yp-yv)**2.0))  #MSE

			# #WRITE TO SCREEN
			# if(iteration==0):    print("iteration	training_loss	validation_loss") 
			# if(iteration%25==0): print(iteration,"	",training_loss,"	",validation_loss) 
			
			#RECORD FOR PLOTING
			loss_train.append(MSE_T); 
			loss_val.append(MSE_V)
			iterations.append(iteration); iteration+=1


	
			df=np.mean(np.absolute(f(xip1,xt2,yt2)-f(xi,xt2,yt2)))
			#print(iteration,"	",xi,"	","	",f(xi,xt2,yt2)) #,df) 

			if(df<tol):
				print("STOPPING CRITERION MET (STOPPING TRAINING)")
				break

		#UPDATE FOR NEXT ITERATION OF LOOP
		xi=xip1
		iteration=iteration+1

	return xi

popt=optimizer(loss,po)


#PREDICTIONS
xm=np.array(sorted(xt))
yp=np.array(model(xm,popt))

#UN-NORMALIZE
def unnorm_x(x): 
	return XSTD*x+XMEAN  
def unnorm_y(y): 
	return YSTD*y+YMEAN 

#FUNCTION PLOTS
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(unnorm_x(xt), unnorm_y(yt), 'o', label='Training set')
	ax.plot(unnorm_x(xv), unnorm_y(yv), 'x', label='Validation set')
	ax.plot(unnorm_x(xm),unnorm_y(yp), '-', label='Model')
	plt.xlabel('x', fontsize=18)
	plt.ylabel('y', fontsize=18)
	plt.legend()
	plt.show()

#PARITY PLOTS
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(model(xt,popt), yt, 'o', label='Training set')
	ax.plot(model(xv,popt), yv, 'o', label='Validation set')
	plt.xlabel('y predicted', fontsize=18)
	plt.ylabel('y data', fontsize=18)
	plt.legend()
	plt.show()

#MONITOR TRAINING AND VALIDATION LOSS  
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(iterations, loss_train, 'o', label='Training loss')
	ax.plot(iterations, loss_val, 'o', label='Validation loss')
	plt.xlabel('optimizer iterations', fontsize=18)
	plt.ylabel('loss', fontsize=18)
	plt.legend()
	plt.show()