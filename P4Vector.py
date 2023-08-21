import numpy as np
import tensorflow as tf

import ops


class P4Vector(object):
    def __init__(self,name):
        self._name = name
        
    def name(self):
        return self._name
        
    def setName(self,name):
        self._name = name
    
    def px(self):
        raise NotImplementedError
        
    def py(self):
        raise NotImplementedError
        
    def pz(self):
        raise NotImplementedError
        
    def energy(self):
        raise NotImplementedError
        
    def mass2(self):
        raise NotImplementedError
        
    def mass(self):
        raise NotImplementedError
        
    def p2(self):
        raise NotImplementedError
        
    def p(self):
        raise NotImplementedError
    
    def pt2(self):
        raise NotImplementedError
    
    def pt(self):
        raise NotImplementedError
        
    def eta(self):
        raise NotImplementedError
        
    def phi(self):
        raise NotImplementedError
        
    def tensor(self):
        raise NotImplementedError
          
class PxPyPzEVector(P4Vector):
    def __init__(self,name,px,py,pz,energy,tensor=None):
        P4Vector.__init__(self,name)
        self._px = px
        self._py = py
        self._pz = pz
        self._energy = energy
        self._tensor=tensor
        
    @staticmethod
    def makeTensor(name,nbatch=1):
        tensor = tf.Variable(
            initial_value=np.zeros((nbatch,4),dtype=np.float32),
            trainable=True,
            name=name,
            dtype=tf.float32,
        )
        value = ops.Constant(tensor)
        
        return PxPyPzEVector(
            name,
            ops.slicev(value,0),
            ops.slicev(value,1),
            ops.slicev(value,2),
            ops.slicev(value,3),
            tensor=tensor
        )
        
    def setPxPyPzE(self,arr):
        if self._tensor is None:
            raise Exception("Cannot assign value to vector since '"+str(self._name)+"' not created from tensor")
        if self._tensor.shape!=arr.shape:
            raise Exception("Vector has different dimensions '"+str(self._tensor.shape)+"' than value '"+str(arr.shape)+"'")
        self._tensor.assign(arr)
        
    def tensor(self):
        return self._tensor
        
    def px(self):
        return self._px
        
    def py(self):
        return self._py
        
    def pz(self):
        return self._pz
        
    def energy(self):
        return self._energy
        
    def mass2(self):
        return ops.UnaryOp(tf.nn.relu,ops.square(self._energy)-self.p2()+ops.Constant(1e-8))
        
    def mass(self):
        return ops.sqrt(self.mass2())
        
    def p2(self):
        return self.pt2()+ops.square(self._pz)
        
    def p(self):
        return ops.sqrt(self.p2())
    
    def pt2(self):
        return ops.square(self._px)+ops.square(self._py)
   
    def pt(self):
        return ops.sqrt(self.pt2())
        
    def eta(self):
        return ops.atanh(self._pz/self.p())
        
    def phi(self):
        return ops.atan2(self._py,self._px)
        
    def __str__(self):
        return "PxPyPzEVector (%s: px=%.2f,py=%.2f,pz=%.2f,m=%.2f)"%(
            self.name(),self.px()(),self.py()(),self.pz()(),self.mass()()
        )

def P4Vector__add__(self,p4):
    pxSum = self.px()+p4.px()
    pySum = self.py()+p4.py()
    pzSum = self.pz()+p4.pz()
    eSum = self.energy()+p4.energy()
    
    return PxPyPzEVector(
        "("+self.name()+"+"+p4.name()+")",
        pxSum,
        pySum,
        pzSum,
        eSum
    )
    
P4Vector.__add__ = P4Vector__add__
    
def P4Vector__sub__(self,p4):
    pxDiff = self.px()-p4.px()
    pyDiff = self.py()-p4.py()
    pzDiff = self.pz()-p4.pz()
    eDiff = self.energy()-p4.energy()
    return PxPyPzEVector(
        "("+self.name()+"-"+p4.name()+")",
        pxDiff,
        pyDiff,
        pzDiff,
        eDiff
    )
 
P4Vector.__sub__ = P4Vector__sub__


class PtPhiEtaMassVector(P4Vector):
    def __init__(self,name,pt,phi,eta,mass,tensor=None):
        P4Vector.__init__(self,name)
        self._pt = pt
        self._phi = phi
        self._eta = eta
        self._mass = mass
        self._tensor = tensor
        
    @staticmethod
    def fromTensor(name,tensor):
        if len(tensor.shape)!=2 or tensor.shape[1]!=4:
            raise Exception("Tensor needs to be of shape [<any>,4]")
            
        value = ops.Constant(tensor)
            
        return PtPhiEtaMassVector(
            name,
            ops.slicev(value,0),
            ops.slicev(value,1),
            ops.slicev(value,2),
            ops.slicev(value,3),
            tensor=tensor
        )
        
    @staticmethod
    def makeTensor(name,nbatch=1):
        tensor = tf.Variable(
            initial_value=np.zeros((nbatch,4),dtype=np.float32),
            trainable=True,
            name=name,
            dtype=tf.float32,
        )
        value = ops.Constant(tensor)
        
        return PtPhiEtaMassVector(
            name,
            ops.slicev(value,0),
            ops.slicev(value,1),
            ops.slicev(value,2),
            ops.slicev(value,3),
            tensor=tensor
        )
        
    def setPtPhiEtaMass(self,arr):
        if self._tensor is None:
            raise Exception("Cannot assign value to vector since '"+str(self._name)+"' not created from tensor")
        if self._tensor.shape!=arr.shape:
            raise Exception("Vector has different dimensions '"+str(self._tensor.shape)+"' than value '"+str(arr.shape)+"'")
        self._tensor.assign(arr)
        
    def tensor(self):
        return self._tensor
        
    def px(self):
        return self._pt*ops.cos(self._phi)
        
    def py(self):
        return self._pt*ops.sin(self._phi)
        
    def pz(self):
        return self._pt*ops.sinh(self._eta)
        
    def energy(self):
        return ops.sqrt(self.p2()+self.mass2())
        
    def mass2(self):
        return ops.square(self._mass)
        
    def mass(self):
        return self._mass
        
    def p2(self):
        return self.pt2()+ops.square(self.pz())
        
    def p(self):
        return ops.sqrt(self.p2())
    
    def pt2(self):
        return ops.square(self._pt)
   
    def pt(self):
        return self._pt
        
    def eta(self):
        return self._eta
        
    def phi(self):
        return self._phi

    def __str__(self):
        return "PtPhiEtaMassVector (%s: pt=%.2f,phi=%.2f,eta=%.2f,m=%.2f)"%(
            self.name(),self.pt()(),self.phi()(),self.eta()(),self.mass()()
        )



        
