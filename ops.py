import numpy as np
import tensorflow as tf

class Op(object):
    def __init__(self):
        pass
        
class Constant(Op):
    def __init__(self,value):
        self.value = value
        
    def __call__(self):
        return self.value

class UnaryOp(Op):
    def __init__(self,unaryFct,node):
        self.unaryFct = unaryFct
        if type(node) == type(float()):
            self.node = Constant(node)
        else:
            self.node = node
        
    def __call__(self):
        return self.unaryFct(self.node())
        
sqrt = lambda x: UnaryOp(tf.sqrt,x)
square = lambda x: UnaryOp(tf.square,x)
exp = lambda x: UnaryOp(tf.exp,x)
log = lambda x: UnaryOp(tf.log,x)
sin = lambda x: UnaryOp(tf.sin,x)
cos = lambda x: UnaryOp(tf.cos,x)
tan = lambda x: UnaryOp(tf.tan,x)
asin = lambda x: UnaryOp(tf.asin,x)
acos = lambda x: UnaryOp(tf.acos,x)
atan = lambda x: UnaryOp(tf.atan,x)
sinh = lambda x: UnaryOp(tf.sinh,x)
cosh = lambda x: UnaryOp(tf.cosh,x)
tanh = lambda x: UnaryOp(tf.tanh,x)
asinh = lambda x: UnaryOp(tf.asinh,x)
acosh = lambda x: UnaryOp(tf.acosh,x)
atanh = lambda x: UnaryOp(tf.atanh,x)

reduce_sum = lambda x: UnaryOp(tf.reduce_sum,x)
reduce_mean = lambda x: UnaryOp(tf.reduce_mean,x)

slicev = lambda x,i: UnaryOp(lambda n: n[:,i],x)
    

class BinaryOp(Op):
    def __init__(self,binaryFct,node1,node2):
        self.binaryFct = binaryFct
        if type(node1) == type(float()):
            self.node1 = Constant(node1)
        else:
            self.node1 = node1
            
        if type(node2) == type(float()):
            self.node2 = Constant(node2)
        else:
            self.node2 = node2
        
    def __call__(self):
        return self.binaryFct(self.node1(),self.node2())
        
add = lambda x,y: BinaryOp(lambda n1,n2: n1+n2,x,y)
sub = lambda x,y: BinaryOp(lambda n1,n2: n1-n2,x,y)
mul = lambda x,y: BinaryOp(lambda n1,n2: n1*n2,x,y)
div = lambda x,y: BinaryOp(lambda n1,n2: n1/n2,x,y)

atan2 = lambda x,y: BinaryOp(tf.atan2,x,y)


        
Op.__add__ = add
Op.__sub__ = sub
Op.__mul__ = mul
Op.__truediv__ = div


