import numpy as np

class test(object):
  def __init__(self):
  # put a lambda here
    self.hello = lambda:'Hello World'
    self.activateF = lambda x : 1/(1+np.exp(-x))

myobj = test()
#print myobj.hello()
print(myobj.activateF(1))
