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
        self.Finished = False
        
    def resetGame(self):
        self.__init__()

    '''
    Function for checking winning conditions
    Input will be the player, row & col of move played
    Search for win in the col, row and the two diagonals
    '''
    
    def check_vertical(self, sub_str, col):
        return sub_str in self.state[:, col].astype(str)

    
    def check_horizontal(self, sub_str, row):
        return sub_str in self.state[row, :].astype(str)
    
    def check_diagonal(self, sub_str, row, col):
        left_diagonal = ''

        #first go to the lefmost point in the left diagonal
        i = row - min(row, col)
        j = col - min(row, col)
        while i < self.height and j < self.width:
            left_diagonal += f'{self.state[row, col]} '
            i+=1
            j+=1
        
        right_diagonal = ''

        #first go to the rightmost point in the right diagonal
        i  = row - min(row, col)
        j = col + min(row, col)
        while i < self.height and j >= 0:
            right_diagonal += f'{self.state[row, col]} '
            i+=1
            j-=1

        return sub_str in left_diagonal or sub_str in right_diagonal
        

    def check_win(self, player, row, col):
        win_substr = ' '.join([self.players[player]] * 4)
        if self.check_vertical(win_substr, col) or self.check_horizontal(win_substr, row) or self.check_diagonal(win_substr, row, col):
            self.Finished = True
        

    '''
    Function for returning the columns which are not full (the topmost slot in the column should be a 0)
    '''

    def free_cols(self):
        return [col for col in range(self.width) if self.state[0, col] == 0]
    




    

