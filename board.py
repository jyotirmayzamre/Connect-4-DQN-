import numpy as np

'''
Class for the connect 4 game:
Number of rows = 4, Number of columns = 5
The board will be a 2D Numpy array consisting of 0s, 1s, and 2s (where 1 is player 1, 2 is player 2, 0 is an empty slot)
Rewards are as follows: {win: 1, draw: -0.5, lose: -1} (we want to maximize winning)
'''

class connect4:
    def __init__(self):
        self.width = 7
        self.height = 6
        self.state = np.zeros([self.height, self.width], dtype=np.uint8)
        self.players = {'P1': 1, 'P2': 2}
        self.rewards = {'Win': 1, 'Draw': -0.5, 'Lose': -1}
        self.Finished = False
        
    def resetGame(self):
        self.__init__()

    
    '''
    Function for returning the columns which are not full (the topmost slot in the column should be a 0)
    '''

    def free_cols(self):
        return [col for col in range(self.width) if self.state[0, col] == 0]



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

        #first go to the lefmost point in the left diagonal of the row, col
        i = row - min(row, col)
        j = col - min(row, col)
        while i < self.height and j < self.width:
            left_diagonal += f'{self.state[row, col]} '
            i+=1
            j+=1
        
        right_diagonal = ''

        #first go to the rightmost point in the right diagonal of the row, col
        i  = row - min(row, col)
        j = col + min(row, col)
        while i < self.height and j > 0:
            right_diagonal += f'{self.state[row, col]} '
            i+=1
            j-=1

        return sub_str in left_diagonal or sub_str in right_diagonal
    
    #we just need to check if the board is full 
    def is_draw(self):
        for col in range(self.width):
            if self.statep[0][col] == 0:
                return False
        return True

    def check_win(self, player, row, col):
        win_substr = ' '.join([self.players[player]] * 4)
        #if either of the conditions passes, the current player has won
        if self.check_vertical(win_substr, col) or self.check_horizontal(win_substr, row) or self.check_diagonal(win_substr, row, col):
            self.Finished = True
        
        if self.Finished:
            return self.rewards['Win']
        elif self.is_draw():
            return self.rewards['Draw']
        else:
            return 0

    '''
    Function for making a move.
    If the move is valid, drop the token at the lowest empty space in the column
    Once the move is made, check winning conditions

    '''

    def move(self, player, col):
        #check if there is free space in the column
        if self.state[0, col] == 0:
            row = np.where(self.state[:, col])[0][-1]
            self.state[row, col] = self.players[player]
            return self.state.copy(), self.check_win(player, row, col)


        else:
            print('Invalid move')
            return self.state.copy(), 0
            


    
    




    

