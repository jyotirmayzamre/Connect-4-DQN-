#Simulate a game against the trained bot
import torch
from board import C4
from net import DQN

env = C4()
model = DQN()
model.load_state_dict(torch.load('C4(novel).pth'))
p1_turn = True
last = 'P1'
env.disp()
while not env.Finished:
    if p1_turn:
        last = 'P1'
        col = int(input("Enter a column number: "))
        env.move('P1', col)
        env.disp()    
        p1_turn = not p1_turn

    else:
        last = 'P2'
        available = env.free_cols()
        state = env.state.copy()
        state = torch.tensor(state, dtype=torch.float).view(1, 1, *state.shape)
        with torch.no_grad():
            actions = model(state)[0, :]
            vals = [actions[i] for i in available]
            move = available[np.argmax(vals)]
        
        env.move('P2', move)
        env.disp()
        p1_turn = not p1_turn
if last == 'P1':
    print('You have won against the bot!')
else:
    print('The bot has won against you!')