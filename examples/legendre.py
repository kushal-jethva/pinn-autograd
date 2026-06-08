from PINN import PINN,Var
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

# 5th order legendre polynomial
def legendre(model, x):
    # ODE: (1-x^2) * u_xx - 2*x*u_x + n(n+1)*u  = 0
    u = model.forward(x)
    u_x = u.diff(x)
    u_xx = u_x.diff(x)
    
    one=Var(1)
    res = (one-x**2)*u_xx - x*u_x*2 + u*(5)*(5+1)
    return res**2

#Boundary Condition /Data
x = Var(np.linspace(-1,1,4), 'x')

# solution at boundary condition
y = Var((1/8)*(63*x.data**5 - 70*x.data**3 + 15*x.data),'y')

#Collocation points 
x_coll = Var(np.linspace(-1, 1, 100),'x collocation')

# solution at collocation points
y_coll = (1/8)*(63*x_coll.data**5 - 70*x_coll.data**3 + 15*x_coll.data)



# PINN model
pinn_model = PINN(layers=[1,20,16,16,1], act=['softplus','tanh','tanh'], initializer=PINN.Xavier)

# Training
while True:
    pinn_model.train(legendre, x, y, x_coll, epochs=10, lr=0.01, lambda_p=0.004) # less importance to avoid trivial 0 solution
    if pinn_model.loss<5e-4:break
    

# NN model
nn_model= PINN(layers=[1,20,16,16,1], act=['softplus','tanh','tanh'], initializer=PINN.Xavier)
                    
# Training NN for same no. of epochs as PINN
nn_model.train_nn(x, y, epochs=pinn_model.epoch, lr=0.01)



# Plot
plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)
plt.scatter(x.data, y.data, color='red', label='Training Data')
plt.plot(x_coll.data,y_coll , 'k--', alpha=0.5, label='True Physics')
plt.plot(x_coll.data, pinn_model.forward(x_coll).data, 'b-', label='PINN Prediction')
plt.title("PINN (Physics Informed)")
plt.legend()


plt.subplot(1, 2, 2)
plt.scatter(x.data, y.data, color='red', label='Training Data')
plt.plot(x_coll.data, y_coll, 'k--', alpha=0.5, label='True Physics')
plt.plot(x_coll.data, nn_model.forward(x_coll).data, 'g-', label='Standard NN')
plt.title("Standard NN (Data Only)")
plt.legend()

plt.show()
