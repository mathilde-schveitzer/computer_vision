import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

def focal_loss(p,gamma):
  return -np.log(p)*(1-p)**gamma

pt=np.linspace(0,1,50)
pt=pt[1::]
gamma=[0,0.5,1,2,5]
fig=plt.figure()

for gam in gamma : 
  plt.plot(pt, focal_loss(pt,gam), label='$\gamma={}$'.format(gam))

plt.title("Focal Loss")
plt.xlabel("Probability of ground truth class")
plt.legend(loc="upper right")