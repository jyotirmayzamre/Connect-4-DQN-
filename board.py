import numpy as np

'''
Class for the connect 4 game:
Number of rows = 4, Number of columns = 5
The board will be a 2D Numpy array consisting of 0s, 1s, and 2s (where 1 is player 1, 2 is player 2, 0 is an empty slot)
Rewards are as follows: {win: 1, draw: 0, lose: -1} (we want to maximize winning)
'''

class connect4:
    def __init__(self):
        self.width = 5
        self.height = 4
        self.state = np.zeros([self.height, self.width], dtype=np.uint8)
        self.players = {'P1': 1, 'P2': 2}
        self.rewards = {'Win': 1, 'Draw': 0, 'Lose': -1}
        
    def resetGame(self):
        self.__init__()

    '''
    
    '''




    

