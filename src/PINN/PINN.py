from .autograd import Var
import numpy as np
from numpy.random import normal as norm
import numpy.random as r

class PINN:
    '''
    A lightweight, dependency-free (purely numpy) Physics-Informed Neural Network (PINN) framework
    built completely from scratch on top of custom reverse-mode automatic differentiation engine.
    
    extensible with custom optimizer,loss,initializer
    '''
    def __init__(self,layers:list=[], act:list=[],**kwargs):
        """
        Parameters:
        ----------
        
        layers:list[int] = [input,hidden1,hidden2,...,output] # no. of neurons in respective layer 
        act:list[str] = ["ReLU","ReLU"..]  # only for hidden layers
        
        **kwargs:
            load="file_name"
            initializer=callable  #custom weights/biases initializer, default He
            opitmizer=callable    #custom optimizer, default adam
            activation={"name":callable} #custom activation functions
            loss_f=callable->'Var'         #custom loss function, defaul MSE
            save_loss=bool               #save model loss history per epoch, default True
        
        example:
            # 1 input neuron , 2 hidden layer each of 2 neurons with tanh , 1 output neuron
            model=PINN(layers=[1,2,1],act=['tanh'],
                        initializer=PINN.Xavier,
                        optimizer=PINN.sgd)
        
        """
        
        self.initializer = kwargs.get("initializer",PINN.He)
        self.optimizer = kwargs.get("optimizer",PINN.adam)
        self.loss_f=kwargs.get("loss",PINN.MSE)
        self.loss=0
        self.act_f={'ReLU':lambda a: a.ReLU(),
                        'tanh':lambda a: a.tanh(),
                        'softplus':lambda a: a.softplus()
                        }
        self.act_f.update(kwargs.get("activation",{}))
        self.save_loss = kwargs.get("save_loss",True)
        
        
        load_file=kwargs.get("load",None)
        
        if load_file: self.load(load_file)
        else:
            self.layers=layers
            self.act=act
            
            self.weights, self.biases = [], []
            self.loss_history=[]
            
            # For ADAM optimizer
            self.w_m, self.w_v = [], []
            self.b_m, self.b_v = [], []
            
            self.initializer(self)
            self.epoch, self.lr = 0,0
       
    
    # INITIALIZERS 
    @staticmethod
    def Xavier(self):
        neurons=self.layers
        for i in range(1, len(neurons)):
            limit = np.sqrt(6 / (neurons[i-1] + neurons[i]))
            W_val = np.random.uniform(-limit, limit, (neurons[i-1], neurons[i]))
            b_val = np.zeros((1, neurons[i]))
            self._add_layer(W_val, b_val, i)
    
    @staticmethod
    def He(self):
        neurons = self.layers
        for i in range(1, len(neurons)):
            std = np.sqrt(2 / neurons[i-1])  
            W_val = np.random.randn(neurons[i-1], neurons[i]) * std
            b_val = np.zeros((1, neurons[i]))
            self._add_layer(W_val, b_val, i)
    
            
    def _add_layer(self, W_val, b_val, i):
        self.weights.append(Var(W_val, f'W{i}'))
        self.biases.append(Var(b_val, f'b{i}'))
        
        w = np.zeros_like(W_val)
        b = np.zeros_like(b_val)
        self.w_m.append(w.copy())
        self.w_v.append(w.copy())
        self.b_m.append(b.copy())
        self.b_v.append(b.copy())
     
    #OPTIMIZERS
    @staticmethod
    def sgd(self):
        lr = self.lr
        for w in self.weights:
            w.data -= lr * w.grad.data
        for b in self.biases:
            b.data -= lr * b.grad.data
            
    @staticmethod      
    def adam(self):
        lr = self.lr
        t = self.epoch
        b1,b2=0.9,0.999
        for I,w in enumerate(self.weights):
            self.w_m[I]=b1*self.w_m[I] + (1-b1)*w.grad.data
            self.w_v[I]=b2*self.w_v[I] + (1-b2)*(w.grad.data**2)
            
            m= self.w_m[I]/(1-b1**t)
            v= self.w_v[I]/(1-b2**t)
            
            w.data-=(m*lr)/(np.sqrt(v)+1e-8)
            
        for I,b in enumerate(self.biases):
            self.b_m[I]=b1*self.b_m[I] + (1-b1)*b.grad.data
            self.b_v[I]=b2*self.b_v[I] + (1-b2)*(b.grad.data**2)
            
            m= self.b_m[I]/(1-b1**t)
            v= self.b_v[I]/(1-b2**t)
            
            b.data-=(m*lr)/(np.sqrt(v)+1e-8)
            
   
         
    #PREDICT
    def predict(self, x):
        return self.forward(x)
    def forward(self, x):
        a = x
        for i in range(len(self.weights) - 1):
            a = a @ self.weights[i] + self.biases[i]
            a = self.act_f[self.act[i]](a)
            
        return a @ self.weights[-1] + self.biases[-1] # last layer
        
    # LOSS   
    @staticmethod
    def MSE(pred,true)->'Var':
        return ((pred - true) **2).mean()
      
    # PINN training
    def train(self, physics_residual, x0, y0, x_coll, 
                    epochs:int, lr:float, lambda_p:float):
        '''
        PINN network train function
        
        Paramters:
        ---------
        physics_residual: callable->'Var'
            gives the residual of differential equation 
        x0,y0: Var matrices 
            x0 training/boundary dataset , y0 true output corresponding to x0
        x_coll: Var matrix
            collocation point within domain to train PINN
        epochs: int
            no. of epochs to train
        lr: float
            learning rate for the network
        lambda_p: float
            importance given to physics_loss
        '''
        self.lr = lr
        
        for epoch in range(epochs):
            self.epoch+=1
            
            #Boundary Loss
            y_pred0 = self.forward(x0)  
            loss_d = self.loss_f(y_pred0,y0)
            
            #Physics Loss
            phy_res = physics_residual(self, x_coll)
            loss_p = phy_res.mean()
            
            total_loss = loss_d + lambda_p * loss_p
            self.loss=total_loss.data.mean()
            total_loss.backward()
            
            self.optimizer(self)
           
            if self.epoch % 10 == 0:
                print(f"Epoch {self.epoch} | Boundary Loss: {loss_d.data.item():.4e} | Physics Loss: {loss_p.data.item():.4e}")
            
            if self.save_loss:self.loss_history.append(total_loss.data.item())
     
    # Standard Neural Network training
    def train_nn(self, x, y_true, epochs:int, lr:float):
        '''
        Standard neural netowrk
        
        Parameters:
        ----------
        x: Var matrix
            training data
        y_true: Var matrix
            true output corresponding to x
        epochs: int
            number of epochs to train
        lr: float
            learning rate for the network
        '''
        self.lr=lr
        for epoch in range(epochs):
            self.epoch+=1
            y_pred = self.forward(x)
            
            loss = self.loss_f(y_pred,y_true)
            self.loss=loss.data.mean()
            loss.backward()
            
            self.optimizer(self)
           
           
            if self.epoch % 10 == 0:
                print(f"Epoch {self.epoch} | Loss: {loss.data.item():.4e} ")
                
            if self.save_loss:self.loss_history.append(loss.data.item())
            
            
    # SAVE/LOAD models  
    def save(self,filename:str):
        np.savez(f"{filename}.npz",
         layers=np.array(self.layers),
         act = np.array(self.act),
         save_loss = self.save_loss,
         weights=np.array([w.data for w in self.weights], dtype=object),
         biases=np.array([b.data.reshape(-1) for b in self.biases], dtype=object),
         loss_history=self.loss_history,
         epoch = self.epoch,
         lr = self.lr,
         w_m=np.array(self.w_m,dtype=object),
         w_v=np.array(self.w_v,dtype=object) ,
         b_m=np.array([np.array(b).reshape(-1) for b in self.b_m],dtype=object),
         b_v=np.array([np.array(b).reshape(-1) for b in self.b_v],dtype=object))
         
    def load(self, filename: str):
        data = np.load(f"{filename}.npz", allow_pickle=True)
        self.layers = data["layers"].tolist()
        self.act = data['act'].tolist()     
        self.save_loss = bool(data['save_loss'])
        
        bias=data['biases']
        bias_v=data['b_v']
        bias_m=data['b_m']
        b=0
        neurons = self.layers
        for i in range(1, len(self.layers)):
            bias[b] = bias[b].reshape(1, neurons[i])
            bias_v[b] = bias_v[b].reshape(1, neurons[i])
            bias_m[b] = bias_m[b].reshape(1, neurons[i])
            b+=1
            
        self.weights = [Var(w, f'W{i+1}') for i,w in enumerate(data['weights'])]
        self.biases  = [Var(b, f'b{i+1}') for i,b in enumerate(bias)]
        self.w_v = data['w_v']
        self.w_m =  data['w_m']
        self.b_m = [b for b in bias_m]
        self.b_v = [b for b in bias_v]
        self.loss_history = data['loss_history'].tolist()
        self.epoch = int(data['epoch'])
        self.lr = float(data['lr'])
