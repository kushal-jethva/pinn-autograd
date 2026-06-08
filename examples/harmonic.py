from PINN import PINN,Var
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

# Damped Harmonic Oscillator
def harmonic_oscillator_physics(model, t):
    # ODE: u'' + 0.2u' + u = 0
    
    u = model.forward(t)
    u_t = u.diff(t)
    u_tt = u_t.diff(t)
    
    res = u_tt + Var(np.array([[0.2]])) * u_t + u
    return res**2

# Boundary Condition / Data
t = Var(np.array([0, 2, 4, 6, 8]),'time')
# True: exp(-0.1t) * cos(t)
u = Var(np.exp(-0.1 * t.data) * np.cos(t.data),'u')


# Collocation points 
t_coll = Var(np.linspace(0, 10, 200),'t collocation')
# solution at collocation points
u_coll = np.exp(-0.1 * t_coll.data) * np.cos(t_coll.data)

# PINN model
pinn_model = PINN(layers=[1, 20,20, 1],act=['tanh','tanh'],initializer=PINN.Xavier)

while True:
    pinn_model.train(harmonic_oscillator_physics, t, u, t_coll, epochs=10, lr=0.01, lambda_p=2)
    if pinn_model.loss<1e-4:break
    
# NN Model
nn_model = PINN(layers=[1, 20,20, 1],act=['tanh','tanh'],initializer=PINN.Xavier)

# Training NN for same no. of epochs as PINN
nn_model.train_nn(t, u, epochs=pinn_model.epoch, lr=0.01)

# Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(t.data, u.data, color='red', label='Training Data')
plt.plot(t_coll.data, u_coll, 'k--', alpha=0.5, label='True Physics')
plt.plot(t_coll.data, pinn_model.forward(t_coll).data, 'b-', label='PINN Prediction')
plt.title("PINN (Physics Informed)")
plt.legend()


plt.subplot(1, 2, 2)
plt.scatter(t.data, u.data, color='red', label='Training Data')
plt.plot(t_coll.data, u_coll, 'k--', alpha=0.5, label='True Physics')
plt.plot(t_coll.data, nn_model.forward(t_coll).data, 'g-', label='Standard NN')
plt.title("Standard NN (Data Only)")
plt.legend()

plt.show()
