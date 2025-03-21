'''
Experience Replay will be used to train.
Any transition that is observed will be stored: (state, action taken, reward received, next state)
We can randomly sample from this list to use for training instead of training on each state-action pair
'''

from random import sample

class Expr_Replay:
    def __init__(self):
        self.store = []

    def sample(self, num):
        return sample(self.store, num)
    
    def add(self, transition):
        self.store.append(transition)