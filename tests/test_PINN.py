import os
import numpy as np
from PINN import PINN,Var


def close(c,other): assert np.allclose(c.data,other)

def test_init():
    model = PINN([1,2,1],['ReLU'])
    assert np.allclose(model.weights[0].data.shape,(1,2))
    assert np.allclose(model.weights[1].data.shape,(2,1))

def test_save_load():
    x=Var([0,1,2,3,4,5])
    y=3*x
    model = PINN([1,2,1],['ReLU'])
    model.train_nn(x,y,epochs=10,lr=0.01)
    
    model.save('test_model')
    
    new=PINN(load='test_model')
    
    
    assert np.array_equal(model.layers,new.layers)
    assert np.array_equal(model.act,new.act)
    assert np.allclose(model.lr,new.lr)
    assert np.allclose(model.epoch,new.epoch)
    assert np.array_equal(model.loss_history,new.loss_history)
    for i,j in zip(model.w_v,new.w_v):assert np.array_equal(i,j)
    for i,j in zip(model.w_m,new.w_m):assert np.array_equal(i,j)
    for i,j in zip(model.b_m,new.b_m):assert np.array_equal(i,j)
    for i,j in zip(model.b_v,new.b_v):assert np.array_equal(i,j)

    assert np.array_equal(model.predict(x).data,new.predict(x).data)
    
    os.remove('test_model.npz')

def test_forward():
    model = PINN([3, 8, 8, 1], ['tanh', 'ReLU'])
    x = Var(np.random.randn(10, 3))
    out = model.forward(x)
    assert out.data.shape == (10, 1)
    
def test_train_loss_decreases():
    model = PINN([1, 8, 1], ['tanh'])
    x = Var(np.linspace(0, 1, 20).reshape(-1, 1))
    y = Var(np.sin(x.data))
    initial = ((model.forward(x) - y) ** 2).mean().data.item()
    model.train_nn(x, y, epochs=200, lr=0.01)
    final = ((model.forward(x) - y) ** 2).mean().data.item()
    assert final < initial

    
