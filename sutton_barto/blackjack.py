import numpy as np
from ...core import Environment

deck = [1,2,3,4,5,6,7,8,9,10,10,10,10]

def player_reward(a,b):
    return int((a>b))-int((a<b))

def draw_card():
    return np.random.choice(deck)

def draw_hand():
    return [draw_card(), draw_card()]

def usable_ace(hand):
    return 1 in hand and sum(hand)+10 <= 21

def sum_hand(hand):
    # Return current hand total
    if usable_ace(hand):
        return sum(hand)+10
    return sum(hand)

def is_bust(hand): # is this a loosing hand?
    return sum_hand(hand)>21

def score(hand):
    return 0 if is_bust(hand) else sum_hand(hand)

def is_natural(hand):
    return sorted(hand) == [1,10]

class BlackjackEnv(Environment):
    def __init__(self, natural = False):
        self.action_space = [0,1]
        self.natural = natural

    def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player))            
    
    def reset(self):
        self.dealer = draw_hand()
        self.player = draw_hand()

        # If score less than 12, auto-draw
        while sum_hand(self.player) < 12:
            self.player.append(draw_card())

        return self._get_obs()


    def step(self, action):
        assert action in self.action_space

        if action: # hit: add a card to players hand
            self.player.append(draw_card())
            if is_bust(self.player):
                done = True
                reward = -1
            else: 
                done = False
                reward = 0 

        else: #stick: play out the dealers hand and score
            done = True
            while sum_hand(self.dealer)<17:
                self.dealer.append(draw_card())
            reward = player_reward(score(self.player), score(self.dealer))
        
            if self.natural and is_natural(self.player) and reward==1:
                reward = 1.5
        
        return self._get_obs(), reward, done, {}
        

    
