import os
import numpy as np
from PINN import Var

def class_test(c): assert isinstance(c,Var)
def close(c,other): assert np.allclose(c.data,other)

def test_add():
    a,b=Var(1),Var(2)
    c=a+b
    class_test(c)
    close(c,3)
    close(c.diff(a),1)
    close(c.diff(b),1)
    close(c.diff(b).diff(b),0)
    close(c.diff(a).diff(a),0)
    c=2+b
    class_test(c)
    close(c,4)
    
    c+=1
    class_test(c)
    close(c,5)
    
def test_subtract():
    a,b=Var(1),Var(2)
    c=a-b
    class_test(c)
    close(c,-1)
    close(c.diff(a),1)
    close(c.diff(b),-1)
    close(c.diff(b).diff(b),0)
    close(c.diff(a).diff(a),0)
    
    c=2-b
    class_test(c)
    close(c,0)
    
    c-=1
    class_test(c)
    close(c,-1)
    
def test_mul():
    a,b=Var(2),Var(6)
    c=a*b
    class_test(c)
    close(c,12)
    close(c.diff(a),6)
    close(c.diff(b),2)
    close(c.diff(b).diff(b),0)
    close(c.diff(a).diff(a),0)
    
    c=2*a
    class_test(c)
    close(c,4)
    c*=2
    close(c,8)
    
def test_pow():
    b=Var(2)
    a=b**2
    class_test(a)
    close(a,4)
    close(a.diff(b),4)
    close(a.diff(b).diff(b),2)
    a=Var(2)**-1
    close(a,0.5)
  
def test_matmul():
    a=Var([[1,0],[0,1]])
    b=Var([[1,0],[0,1]])
    c=a*b
    class_test(c)
    close(c,[[1,0],[0,1]])
    close(c.diff(b),[[1,0],[0,1]])
    close(c.diff(a),[[1,0],[0,1]])
    close(c.diff(b).diff(b),[[0,0],[0,0]])
    close(c.diff(a).diff(a),[[0,0],[0,0]])
def test_div():
    a,b=Var(2),Var(1)
    c=b/a
    class_test(c)
    close(c,0.5)
    close(c.diff(a),-0.25)
    close(c.diff(b),0.5)
    close(c.diff(a).diff(a),0.25)
    close(c.diff(b).diff(b),0)
def test_neg():
    a=Var(2)
    b=-a
    class_test(b)
    close(b,-2)
    close(b.diff(a),-1)
    close(b.diff(a).diff(a),0)
    
def test_T():
    a=Var([[1,0],[0,1]])
    b=a.T()
    class_test(b)
    close(b,[[1,0],[0,1]])
    close(b.diff(a),1)
    close(b.diff(a).diff(a),0)
def test_exp():
    a=Var(2)
    b=a.exp()
    class_test(b)
    close(b,np.exp(2))
    close(b.diff(a),np.exp(2))
    close(b.diff(a).diff(a),np.exp(2))

def test_mean():
    # consider [x,y..]/data.size = mean
    # here (x+y)/2 = mean | 2*mean = x + y 
    # 2* dmean/dx = 1 =>0.5 and 2*dmean/dy = 1 => 0.5 in matrix [0.5,0.5]
    
    a=Var([1,2])
    b=a.mean()
    class_test(b)
    close(b,1.5)
    close(b.diff(a),[[0.5],[0.5]])

def test_ReLU():
    a=Var(1)
    b=a.ReLU()
    class_test(b)
    close(b,1)
    close(b.diff(a),1)
    close(b.diff(a).diff(a),0)
    
    a=Var(-1)
    b=a.ReLU()
    class_test(b)
    close(b,0)
    close(b.diff(a),0)
    close(b.diff(a).diff(a),0)
    
def test_tanh():
    a=Var(np.pi)
    b=a.tanh()
    class_test(b)
    close(b,np.tanh(np.pi))
    close(b.diff(a),1-np.tanh(np.pi)**2)
    close(b.diff(a).diff(a),-2*np.tanh(np.pi)*(1-np.tanh(np.pi)**2))
def test_softplus():
    a=Var(1)
    b=a.softplus()
    class_test(b)
    close(b,np.log1p(np.exp(1)))
    close(b.diff(a),1/(1+np.exp(-1)))
    close(b.diff(a).diff(a),np.exp(-1)*(1+np.exp(-1))**-2)

def test_sin():
    a=Var(np.pi)
    b=a.sin()
    class_test(b)
    close(b,np.sin(np.pi))
    close(b.diff(a),np.cos(np.pi))
    close(b.diff(a).diff(a),-np.sin(np.pi))

def test_cos():
    a=Var(np.pi)
    b=a.cos()
    class_test(b)
    close(b,np.cos(np.pi))
    close(b.diff(a),-np.sin(np.pi))
    close(b.diff(a).diff(a),-np.cos(np.pi))
   
def test_slice():
    a=Var([[1,2,3],[1,2,3]])
    b=a.slice(0,2)
    class_test(b)
    close(b,[[1,2],[1,2]])
    close(b.diff(a),[[1,1,0],[1,1,0]])
    close(b.diff(a).diff(a),[[0,0,0],[0,0,0]])

def test_concat():
    a=Var([[2],[2]])
    b=Var([[3],[3]])
    c=a.concat(b)
    c=c**2
    class_test(c)
    close(c,[[4,9],[4,9]])
    close(c.diff(a),[[4],[4]])
    close(c.diff(b),[[6],[6]])

def test_stack():
    a=Var([1,1,1])
    b=Var([2,2,2])
    c=Var([3,3,3])
    d=Var.stack(a,b,c)
    d=d**2
    class_test(d)
    close(d,[[1,4,9],[1,4,9],[1,4,9]])
    close(d.diff(a),[[2],[2],[2]])
    close(d.diff(b),[[4],[4],[4]])
    close(d.diff(c),[[6],[6],[6]])
def test_sort():
    # diamond graph
    a=Var(1)
    b=Var(2)
    c=Var(3)
    
    d=a+b
    e=a+c
    
    f = d+e
    f.sort()
    for i,j in zip(f.sorted,[f,e,c,d,b,a]):
        assert j is i


    
    
    
    
   
    
