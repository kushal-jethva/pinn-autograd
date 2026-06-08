from PINN import PINN,Var
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

# Nuclear Decay
def decay(model,t):
    # ODE: dN/dt + 3*N = 0
    N=model.forward(t)
    N_t=N.diff(t)
    res=N_t + 3*N
    return res**2

# Boundary Condition/ Data
t=Var(data = np.array([0,2,4]), name = 't')
# True data at boundary condition
N=Var(data = 10*np.exp(-3*t.data), name = 'N')

# Collocation points
t_coll=Var(data = np.linspace(0,4,500), name = 'collocation t')
# True data at Collocation points
N_coll=10*np.exp(-3*t_coll.data)



#PINN model
pinn_model=PINN(layers=[1,16,16,1],act=['tanh','tanh'],initializer=PINN.Xavier)

while True:
    pinn_model.train(decay,t,N,t_coll,epochs=10,lr=0.01,lambda_p=0.90)
    if pinn_model.loss<1e-4:break
    
    
# NN model
nn_model=PINN(layers=[1,16,16,1],act=['tanh','tanh'],initializer=PINN.Xavier)

# Training NN for same no. of epochs as PINN
nn_model.train_nn(t,N,epochs=pinn_model.epoch,lr=0.01)
    

#plot
plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)
plt.scatter(t.data, N.data, color='red', label='Training Data')
plt.plot(t_coll.data, N_coll, 'k--', alpha=0.5, label='True Physics')
plt.plot(t_coll.data, pinn_model.forward(t_coll).data, 'b-', label='PINN Prediction')
plt.title("PINN (Physics Informed)")
plt.legend()  



plt.subplot(1, 2, 2)
plt.scatter(t.data, N.data, color='red', label='Training Data')
plt.plot(t_coll.data, N_coll, 'k--', alpha=0.5, label='True Physics')
plt.plot(t_coll.data, nn_model.forward(t_coll).data, 'g-', label='Standard NN')
plt.title("Standard NN (Data Only)")
plt.legend()

plt.show()
