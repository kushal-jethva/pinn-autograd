import numpy as np

class Var():
    '''
    autograd engine
    
    - creates computational graph during forward pass
    - stores local gradient
    - differentiates by multiplying local gradients (chain rule)
    - supports matrices and higher order derivatives
    
    Can build custom function with bifunc and ufunc
    '''
    def __init__(self, data = 0,name:str = 'x'):
        '''
        initializing Var object
        
        Parameters:
        ----------
        data: int,float,list[int|float],numpy.ndarray
        name: str (optional)
        
        '''
        if type(data)!=np.ndarray:
            data=np.array(data)
        if data.ndim < 2: data = data.reshape(-1,1)
            
        self.data=data
        self.name=name
        self.child=[]
        self.grad=None # gradeint
        self.topologically_sorted=False
        self.differentiated=False
    
    def __repr__(self): return f"{self.name}:data = {self.data}"
    
    
    # BINARY FUNCTION
    
    def bifunc(self,other,forward,backward_self,backward_other)->'Var':
        """
        creates binary function
        
        Parameters:
        ----------
        self,other:Var
        forward: function(self,other):
	             binary operation on self.data, other.data
	             
        backward_self: function(self,other,parent) -> Var :
		         returns (local gradient of self) * (parent.grad) 
		         
        backward_other: function(self,other,parent) -> Var :
	             returns (local gradient of other) * (parent.grad)
        
        > parent.grad (upstream parent gradient) is required to follow the chain rule during backward pass
        
        Returns:
        -------
            parent: Var
                output after performing operation on self,other
        
        Example:
        -------
            >>> def mul(var1, var2):
            ...     # x = v1 * v2 , dx/dv1 = v2 , dx/dv2 = v1
            ...     return var1.bifunc(
            ...         var2,
            ...         forward = lambda v1, v2: v1.data * v2.data,
            ...         backward_self = lambda v1, v2, x: v2 * x.grad,
            ...         backward_other = lambda v1, v2, x: v1 * x.grad)
            >>> data = mul(Var(4), Var(2)) # 4*2
        """
        if not isinstance(other,Var): other = Var(other)
        p=Var()
        p.data = forward(self,other)
        p.child.extend([self,other])
        
        def _backward():

            local_self_grad = backward_self(self,other,p)
            local_other_grad = backward_other(self,other,p)
            
            if not isinstance(local_self_grad,Var): local_self_grad=Var(local_self_grad)
            if not isinstance(local_other_grad,Var): local_other_grad=Var(local_other_grad)

            self.grad+=local_self_grad
            other.grad+=local_other_grad
        
        p._backward = _backward
        
        return p
    
    
    def __add__(self,other):
        return self.bifunc(
            other,
            forward = lambda s,o: s.data + o.data,
            backward_self = lambda s,o,p: p.grad,
            backward_other = lambda s,o,p: Var(np.sum(p.grad.data, axis=0, keepdims=True)) 
                         if p.grad.data.shape != o.data.shape 
                         else p.grad
        )
        
       
        
    def __iadd__(self,other):return self + other
    def __radd__(self,other):return self + other
        
    def __sub__(self,other):
        p= self.bifunc(
            other,
            forward = lambda s,o: s.data - o.data,
            backward_self = lambda s,o,p: 1*p.grad,
            backward_other = lambda s, o, p: Var(-np.sum(p.grad.data, axis=0, keepdims=True))
                 if p.grad.data.shape != o.data.shape
                 else -p.grad
        )
        return p
        
    def __isub__(self,other):return self - other
    def __rsub__(self,other):
        if not isinstance(other,Var):other=Var(other)
        return other - self
    
    def __mul__(self,other):
        p=self.bifunc(
            other,
            forward = lambda s,o: s.data*o.data,
            backward_self = lambda s,o,p: o*p.grad,
            backward_other = lambda s,o,p: s*p.grad
            )
        return p
        
    def __rmul__(self,other):
        if not isinstance(other,Var):other=Var(other)
        return other*self
    def __imul__(self,other):return self*other
        
    def __pow__(self,other):
        p=self.bifunc(
            other,
            forward = lambda s,o: s.data.astype(float)**o.data,
            backward_self = lambda s,o,p: o*s**(o-1)*p.grad,
            backward_other = lambda s,o,p: Var(np.zeros_like(o.data))
            )
        return p
        
    def __matmul__(self,other):
        p=self.bifunc(
            other,
            forward = lambda s,o: s.data @ o.data,
            backward_self = lambda s,o,p : p.grad @ o.T(),
            backward_other = lambda s,o,p : s.T() @ p.grad
            )
        return p
    def __truediv__(self,other):
        return self * other **-1 
        
    ## UNARY OPERATIONS
    def ufunc(self, forward, backward) -> 'Var':
        """
        creates unary function
        
        Parameters:
        ----------
        self: Var
        forward:  function(data) -> numpy.ndarray|list|int|float : 
				    apply forward function on self.data
				    
        backward: function(self,parent) -> Var :
			        returns (local gradeint of self) * (parent.grad)
        
        > parent.grad (upstream parent gradient) is required to follow the chain rule during backward pass
        
        Returns:
        -------
            parent: Var
                output after performing operation on self
         
        Example:
        -------
            >>> # x = exp(v1) , dx/dv1 = x
            >>> exp = lambda x: x.ufunc(
            ...               forward = lambda v1 : np.exp(v1.data), 
            ...                backward = lambda v1,x: x * x.grad)
            >>> data = exp(Var(42))  
        """
        p = Var()
        p.data = forward(self)
        p.child.append(self)
        
        def _backward():
        
            local_grad = backward(self,p)
            if not isinstance(local_grad, Var): local_grad=Var(local_grad)
            
            self.grad += local_grad

        p._backward = _backward
        return p
    
    def __neg__(self):
        return self.ufunc(
            forward = lambda s: -s.data,
            backward = lambda s,p: -np.ones_like(s)*p.grad)
    def T(self):
        return self.ufunc(
            forward = lambda s:s.data.T,
            backward = lambda s,p: p.grad.T()
        )
        
    
    def exp(self):
        return self.ufunc(
            forward = lambda s:np.exp(s.data),
            backward = lambda s,p: p*p.grad
            )
       
        
    def mean(self):
        return self.ufunc(
            forward = lambda s: np.mean(s.data),
            backward =  lambda s,p: Var(np.ones_like(self.data) / self.data.size) * p.grad
            )
        
    
    
    def ReLU(self):
        return self.ufunc(
            forward = lambda s: np.maximum(0,s.data),
            backward = lambda s,p: Var((s.data > 0).astype(s.data.dtype)) * p.grad
        )
      
   
    def tanh(self):
        # backward 1-tanh**2
        return self.ufunc(
            forward = lambda s: np.tanh(s.data),
            backward = lambda s,p: (Var(np.ones_like(p.data),'1') - p**2 )* p.grad
        )
        
     
    def softplus(self):
        # backward 1 / (1 + exp(-x))
        return self.ufunc(
            forward = lambda s: np.log1p(np.exp(s.data)),
            backward = lambda s,p: ((Var(np.ones_like(p.data),'1') + (-self).exp())**-1) * p.grad
        )
        
    
     
    def sin(self):
        return self.ufunc(
            forward = lambda s : np.sin(s.data),
            backward = lambda s,p: s.cos() * p.grad
            )
       
        
    def cos(self):
        return self.ufunc(
            forward = lambda s : np.cos(s.data),
            backward = lambda s,p: -s.sin() * p.grad
            )
        
    
        
    
    ## FOR PARTIAL DIFF
    def slice(self, start:int, end:int) -> 'Var':
        '''
        extract columns start:end
        
        Parameters:
        ----------
        self: Var array_like
        start: int -> start index of slice
        end: int -> end index of slice (excluded)
        
        return:
        ------
        parent: Var
            start to end indexed columns matrix
        Example:
        -------
        >>> # domain tracking shape: (100, 3) [x_col, y_col, t_col]
        >>> domain = Var(domain_data)
        >>> x = domain.slice(0,2)  # Extracts [x_col,y_col]
        '''
        p = Var(name=f'slice_{start}:{end}')
        p.data = self.data[:, start:end]
        p.child.append(self)

        def _backward():
            grad = np.zeros_like(self.data)
            grad[:, start:end] = p.grad.data 
            self.grad += Var(grad)

        p._backward = _backward
        return p

    def col(self, i:int):
        '''
        extract single column from the Var matrix
        
        Parameters:
        ----------
        self: Var
            matrix containing different feature columns
        i: int
            index of the target column 
        
        Returns:
        -------
        parent: 
            single column Var matrix
        
        Example:
        -------
        >>> # domain tracking shape: (100, 3) [x_col, y_col, t_col]
        >>> domain = Var(domain_data)
        >>> x = domain.col(0)  # Extracts column 0
        >>> y = domain.col(1)  # Extracts column 1 
        >>> t = domain.col(2)  # Extracts column 2 
        '''
        
        return self.slice(i, i+1)  # col is just slice of width 1
    
    def concat(self, other):
        '''
        column-wise stack two Var matrices together
        
        Parameters:
        ----------
        self,other: Var matrices 
                    must have same number of rows = nrows
        Returns:
        -------
        parent: Var([self,other])
        
        Example:
        --------
        >>> # x shape: (100, 1), t shape: (100, 1)
        >>> x = Var(x_data)
        >>> t = Var(t_data)
        >>> domain = x.concat(t)  # [x,t] domain shape (100, 2)
        '''
        split = self.data.shape[1]
        return self.bifunc(
            other,
            forward        = lambda s, o: np.concatenate([s.data, o.data], axis=1),
            backward_self  = lambda s, o, p: p.grad.slice(0, split),
            backward_other = lambda s, o, p: p.grad.slice(split, split + o.data.shape[1])
        )
    
    @staticmethod
    def stack(*vars) -> 'Var':
        '''
        concat multiple Var matrices column-wise
        
        Parameters:
        ----------
        *vars:  multiple Var matrices Var1,Var2,Var3,...
                all must have same number of rows = nrows
        Returns:
        -------
        Var matrix of columns=Var1,Var2,.. and rows = nrows
        
        Example:
        --------
        >>> # x shape: (100, 1),y shape:(100,1), t shape: (100, 1)
        >>> x = Var(x_data)
        >>> y = Var(y_data)
        >>> t = Var(t_data)
        >>> domain = Var.stack(x,y,t)  # [x,y,t] domain shape (100, 3)
        
        '''
        result = vars[0]
        for v in vars[1:]:
            result = result.concat(v)
        return result
        
    
    
    ## TOPOLOGICAL SORT
    def sort(self, nodes=None, visited=None) -> list:
        '''
        Topologically sorting
        
        - start from root node (parent)
        - if children not present in nodes(list) , add them
        - add root node to nodes(list)
        - this is reverse order of the dependencies [...,child2,child1,parent]
        
        This creates a order in which all dependencies(child)
        of parent node are solved(local gradient) first before moving to parent node.
        
        Returns:
        -------
        list
            The ordered list of Var nodes ready for backward.
        '''
    
        if nodes is None: nodes = []
        if visited is None: visited = set()
        
        for c in self.child:
            if c not in visited:
                c.sort(nodes, visited)
                
        visited.add(self)
        nodes.append(self) #root at -1
        self.sorted=nodes[::-1]
        self.topologically_sorted=True
        
        return nodes
    
    ## ACCUMALTING GLOBAL GRADEINT
    def backward(self) -> None:
        '''
        computes gradients
        
        - resets all local gradients
        - computes global gradients
        '''
        if not self.topologically_sorted:self.sort()
        for p in self.sorted: 
            p.grad=Var(np.zeros(p.data.shape))
          
        #starting with parent gradeint = 1 
        self.sorted[0].grad=Var(np.ones_like(self.sorted[0].data))
  
        gradients={}
        
        for p in self.sorted:
            if p.child!=[]: 
                p._backward()       #runs backward on each node
                for c in p.child:   # stores global gradient of each
                    gradients[c]=c.grad

        self.gradients=gradients
        self.differentiated=True
        
       

    def diff(self,other) -> 'Var':
        '''
        gets derivative of self w.r.t. other
        
        Parameters:
        ---------
        self,other: Var 
        
        Example:
        -------
        >>> x = Var(1)
        >>> y = Var(2)
        >>> z = x * y
        >>> print(z.diff(y).data)
        >>> 1.0
        '''
        
        
        if not self.differentiated:self.backward()
          
        return self.gradients.get(other,Var(0,name='0'))
        
    
