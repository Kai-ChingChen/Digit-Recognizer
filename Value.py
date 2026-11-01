import numpy as np
import math

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._prev = tuple(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f'Value(shape = {self.data.shape}, data = {self.data})'

    @staticmethod
    def unbroadcast(grad, shape):
        #sum the gradient back to original input's shape 
        while grad.ndim > len(shape):
            #handle sum over extra dimensions
            grad  = grad.sum(axis = 0)
        for axis, size in enumerate(shape):
            #handle sum if original dimension have size 1
            if size == 1:
                grad = grad.sum(axis = axis, keepdims = True)
        return grad

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += self.unbroadcast(out.grad, self.data.shape)
            other.grad += self.unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += self.unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += self.unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __matmul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        if self.data.ndim != 2 or other.data.ndim != 2:
            raise ValueError("Matrix multiplication required 2D arrays")

        out = Value(self.data @ other.data, (self, other), "@") #(a x b)x(b x c) = (a x c)
        def _backward():
            self.grad += out.grad @ other.data.T #(a x b) = (a x c)x(c x b)
            other.grad += self.data.T @ out.grad #(b x c) = (b x a)x(a x c)

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += out.grad * (1 - t**2)

        out._backward = _backward

        return out

    def __pow__(self, power):
        if not isinstance(power, (int, float)):
            raise TypeError('Power nust be a numeric constant')

        out = Value(self.data ** power, (self, ), "**")
        def _backward():
            self.grad += out.grad * power * (self.data**(power-1))
        out._backward = _backward
        return out

    def exp(self):
        out = Value(np.exp(self.data), (self,), "exp")
        def _backward():
            self.grad += out.grad * out.data
        out._backward = _backward
        return out

    def sum(self):
        out = Value(self.data.sum(), (self,), "sum")
        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out

    def mean(self):
        if self.data.size == 0:
            raise ValueError("Cannot average an empyty array")
        return self.sum() * (1.0/self.data.size)

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other if isinstance(other, Value) else -np.asarray(other))

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * (other**-1)

    def __rtruediv__(self, other):
        return other * (self**-1)

    def backward(self):
        #call backward on each node in reverse topological order
        if self.data.ndim != 0:
            raise ValueError("Loss needs to be scalar")

        topo, visited = [], set()
        def visit(node):
            if node not in visited:
                visited.add(node)
                for parent in node._prev:
                    visit(parent)
                topo.append(node)
        visit(self)
        self.grad[...] = 1.0
        for node in reversed(topo):
            node._backward()
