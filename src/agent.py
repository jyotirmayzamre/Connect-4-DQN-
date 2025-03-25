import torch.optim as optim
from torch import nn
import torch
from board import C4
from net import DQN
from expr import Expr_Replay
import math
from tqdm import tqdm
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self):
        self.learning_rate = 0.001
        self.batch_size = 200
        self.gamma = 0.9
        self.e_max = 1.0
        self.e_min = 0.01
        self.decay_rate = 0.001
        self.env = C4()
        self.memory = Expr_Replay()
        self.main_net = DQN().to(device)
        self.target_net = DQN().to(device)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.optimizer = optim.Adam(self.main_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.steps_done = 0
        self.episodes = 25000
        self.target_lag = 15



    #exponential decay used for epsilon
    def epsilon_decay(self, step):
        return self.e_min + (self.e_max - self.e_min) * math.exp(-self.decay_rate * step)


    #add batch size and channel to make it ready for a conv layer
    def transform_input(self, state):
        return torch.tensor(state, dtype=torch.float, device=device).view(1, 1, *state.shape)

    '''
    Function for carrying out epsilon-greedy strategy for the agent
    '''
    def epsilon_greedy(self, state, free_actions, step):
        state = self.transform_input(state)

        threshold = self.epsilon_decay(step)
            
        #if less then epsilon, then choose a random available action
        #if greater than epsilon, then use the main NN to exploit
        if random.random() < threshold:
            return random.choice(free_actions)
        else:
            actions = self.main_net(state)[0, :]
            vals = [actions[i].detach().cpu().numpy() for i in free_actions]
            return free_actions[np.argmax(vals)]

    '''
    Random Agent that picks a random column
    '''
    def rand_agent(self, free_cols):
        return random.choice(free_cols)


    ''' 
    This is the optimization function for the network
    First sample a batch of transitions from the experience replay
    Separate the 4 components (states, actions, rewards, next_states) into their own tuples and convert into appropriate tensors
    Use the target net to compute max Q values
    Predict Q values using the main net
    Calculate loss and then update weights
    '''
    def optimize(self):
        samples = self.memory.sample(self.batch_size)
        
        #split the transitions in their own batches
        states, actions, rewards, next_states = zip(*samples)

        #transform into tensors
        states = self.transform_input(states, dtype=torch.float, device=device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)

        #create a mask for non terminal states
        mask = tuple(x is not None for x in next_states)
        act_next_states = tuple(x for x in next_states if x is not None)
        act_next_states = self.transform_input(act_next_states, dtype=torch.float, device=device)

        #predicted Q-values
        pred = self.main_net(states).gather(0, actions)

        #target Q-values
        target = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            target[mask] = self.target_net(act_next_states).max(1)[0]
        
        target = (target * self.gamma) + rewards

        loss = self.loss_fn(pred, target.unsqueeze(1))
        
        #backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        

    ''' 
    Function for training our main NN. For each episode, we will randomize who makes the first move so that our agent has experience with both situations.
    Need to rewrite a lot
    '''
    def train(self):
        for i in tqdm(range(self.episodes), desc='Processing'):
            self.env.resetGame()
            moves = 0

            #select random player to go first: P1 is always our agent and P2 is always a random agent
            #exposes our agent to episodes where it starts with the second turn
            first = random.choice(['P1', 'P2'])
            if first == 'P2':
                free_actions = self.env.free_cols()
                a_p2 = self.rand_agent(free_actions)
                s_p2_curr, reward_p2 = self.env.move('P2', a_p2)
                s_p1 = s_p2_curr
            else:
                s_p1 = self.env.state.copy()


            #main loop for each episode
            while True:
                free_actions = self.env.free_cols()
                a_p1 = self.epsilon_greedy(s_p1, free_actions, self.steps_done)
                s_p1_curr, reward_p1 = self.env.move('P1', a_p1)
                self.steps_done += 1
                moves += 1

                #add an experience to the memory store
                if self.env.Finished:
                    self.memory.add([s_p1, a_p1, reward_p1, None])
                    break

                free_actions = self.env.free_cols()
                a_p2 = self.rand_agent(free_actions)
                s_p2_curr, reward_p2 = self.env.move("P2", a_p2)

                #inverse reward is added if the random agent wins
                if self.env.Finished:
                    if reward_p2 == 1:
                        self.memory.add([s_p1, a_p1, -1, None])
                    else:
                        self.memory.add([s_p1, a_p1, -0.5, None])
                    break
                
                #add a small cost for not winning on this move
                #as the number of moves taken increases, increase the cost
                self.memory.add([s_p1, a_p1, -0.1 * moves, s_p2_curr])
                s_p1 = s_p2_curr

                #optimize the main net
                if len(self.memory.store) >= self.batch_size:
                    self.optimize()
            
            #update the weights of the target net every 15 episodes
            if i % self.target_lag == 0:
                self.target_net.load_state_dict(self.main_net.state_dict())

        print('Completed')