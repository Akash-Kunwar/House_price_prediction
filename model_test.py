from joblib import dump, load
import numpy as np 
model=load('Final.joblib')  

inp=np.array([[1.05 ,0 ,8.14 ,0 ,0.537 ,5.935
 ,29.3 ,4.4986 ,4 ,307,21 ,386
 ,6.58]])

print(model.predict(inp))

