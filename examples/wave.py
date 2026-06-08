from PINN import PINN,Var
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def wave_physics(model, xt):
    # PDE: d2u/dt2 - d2u/dx2 = 0 ; c=1
    
    x = xt.col(0)
    t = xt.col(1)
    xt = Var.stack(x,t) # reconstructing to include x,t in computation graph
    u = model.forward(xt)
    
    u_t = u.diff(t)
    u_x = u.diff(x)
    
    u_tt = u_t.diff(t)
    u_xx = u_x.diff(x)
    
    res = u_tt - u_xx
    return res**2

x = np.linspace(0,1,20).reshape(-1,1)
t = np.linspace(0,1,20).reshape(-1,1)

# Boundary condition
x1 = np.ones_like(x) # x=1 
x0 = np.zeros_like(x) # x=0 

xt_bc_1 = np.hstack([x1,t]) # x=1, t=0,1
xt_bc_2 = np.hstack([x0,t]) # x=0, t=0,1

u0 = np.zeros_like(x) # u=0, x= 0 and 1, t=0,1              
t0 = np.zeros_like(t) # time = 0
u_init = np.sin(np.pi * x) # u=sin(x*pi) , x=0,1 , t=0
xt_init =  np.hstack([x,t0]) 

xt_data = np.vstack([xt_init,xt_bc_1,xt_bc_2])
u_data = np.vstack([u_init,u0,u0])


t_mesh, x_mesh = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))

# collocation points
xt_coll = np.hstack([x_mesh.flatten().reshape(-1, 1), 
                        t_mesh.flatten().reshape(-1, 1)])

xt_bc = Var(xt_data)
u_bc = Var(u_data)
xt_coll = Var(xt_coll)

# solution at collocation points
u_coll = np.cos(np.pi*t)*np.sin(np.pi*x)

# PINN model
pinn_model = PINN(layers=[2, 32,16,16, 1],act=['tanh','tanh','tanh'],initializer=PINN.Xavier)

while True:
    pinn_model.train(wave_physics, xt_bc,u_bc , xt_coll, 
                    epochs=10, lr=0.01, lambda_p=0.05)
    if pinn_model.loss<5e-5: break    

# NN model
nn_model = PINN(layers=[2, 32,16,16, 1],act=['tanh','tanh','tanh'],initializer=PINN.Xavier)

# Training NN for same no. of epochs as PINN
nn_model.train_nn(xt_bc,u_bc,epochs=pinn_model.epoch, lr=0.01)




# PLOT
x_plot = np.linspace(0, 1, 20)
t_plot = np.linspace(0, 1, 20)

true_lines = []
for ti in t_plot:
    ui = np.cos(np.pi * ti) * np.sin(np.pi * x_plot)
    true_lines.append((x_plot, np.full_like(x_plot, ti), ui))
    
pinn_lines = []
for ti in t_plot:
    x_var = Var(x_plot.reshape(-1, 1))
    t_var = Var(np.full((20, 1), ti))
    xt_var = Var.stack(x_var, t_var)
    u_pred = pinn_model.forward(xt_var).data[:, 0]
    pinn_lines.append((x_plot, np.full_like(x_plot, ti), u_pred))

nn_lines = []
for ti in t_plot:
    x_var = Var(x_plot.reshape(-1, 1))
    t_var = Var(np.full((20, 1), ti))
    xt_var = Var.stack(x_var, t_var)
    u_pred = nn_model.forward(xt_var).data[:, 0]
    nn_lines.append((x_plot, np.full_like(x_plot, ti), u_pred))
    
    
fig = plt.figure(figsize=(14, 6))

# PINN plot
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.scatter(xt_data[:, 0], xt_data[:, 1], u_data[:, 0],
            color='red', s=40, label='Training Data', zorder=5)

for i, (xi, ti, ui) in enumerate(true_lines):
    ax1.plot(xi, ti, ui, color='black', linestyle='--', alpha=0.5, linewidth=1.5,
             label='True Physics' if i == 0 else "")

for i, (xi, ti, ui) in enumerate(pinn_lines):
    ax1.plot(xi, ti, ui, color='blue', linestyle='-', alpha=0.8, linewidth=1.5,
             label='PINN Prediction' if i == 0 else "")

ax1.set_xlabel('Space (x)', fontsize=11, labelpad=10)
ax1.set_ylabel('Time (t)', fontsize=11, labelpad=10)
ax1.set_zlabel('u(x, t)', fontsize=11, labelpad=10)
ax1.set_title("PINN (Physics Informed)", fontsize=14)
ax1.legend(loc='upper left', fontsize=9)
ax1.view_init(elev=20, azim=40)

# NN plot
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.scatter(xt_data[:, 0], xt_data[:, 1], u_data[:, 0],
            color='red', s=40, label='Training Data', zorder=5)

for i, (xi, ti, ui) in enumerate(true_lines):
    ax2.plot(xi, ti, ui, color='black', linestyle='--', alpha=0.5, linewidth=1.5,
             label='True Physics' if i == 0 else "")

for i, (xi, ti, ui) in enumerate(nn_lines):
    ax2.plot(xi, ti, ui, color='green', linestyle='-', alpha=0.8, linewidth=1.5,
             label='Standard NN' if i == 0 else "")

ax2.set_xlabel('Space (x)', fontsize=11, labelpad=10)
ax2.set_ylabel('Time (t)', fontsize=11, labelpad=10)
ax2.set_zlabel('u(x, t)', fontsize=11, labelpad=10)
ax2.set_title("Standard NN (Data Only)", fontsize=14)
ax2.legend(loc='upper left', fontsize=9)
ax2.view_init(elev=20, azim=40)

plt.tight_layout()
plt.show()
