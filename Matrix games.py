#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 10:07:40 2017

@author: jpmaldonado
"""

import matplotlib.pyplot as plt
import random
import numpy as np

np.set_printoptions(precision=2)

# Base player that chooses a constant mixed strategy every time
class Player:
    def __init__(self, first_move = 1):
        self.my_moves = []
        self.other_moves = []
        self.first_move = first_move
        
    def move(self, strategy):
        # Input: a vector of probability distributions for actions
        # Output: a pure action
        actions = range(len(strategy))
        a = np.random.choice(actions, 1, p=strategy)
        return a
    

if __name__ == "__main__":        

    # Simple matrix game
    payoff_matrix = [[(1,-1),(0,0)], [(0,0), (-1,1)]]
    
    n_rounds = 20
    p1 = Player(first_move = 0)
    p2 = Player(first_move = 1)
    # Random strategy 
    strategy = [0.5, 0.5]
	
    total_p1 = 0.0
    total_p2 = 0.0
    
    plt.ion()
    plt.axis([-0.1,n_rounds+0.1,-0.1,1.1])
    
    tot_payoff_p1 = 0.
    tot_payoff_p2 = 0.
    
    for n in range(1,n_rounds):
        m1 = p1.move(strategy)
        m2 = p2.move(strategy)
        
        # Players update the info of the moves
        p1.my_moves.append(m1)
        p1.other_moves.append(m2)    
        
        p2.my_moves.append(m2)
        p2.other_moves.append(m1)    
        
        ############################################
        ## Show payoffs
        ############################################
        tot_payoff_p1 += payoff_matrix[m1][m2][0]
        tot_payoff_p2 += payoff_matrix[m1][m2][1]
        
        avg_payoff_p1 = round(tot_payoff_p1/n,2)
        avg_payoff_p2 = round(tot_payoff_p2/n,2)
        
        print("-"*20)
        print("Average payoff for player 1: %f " % avg_payoff_p1)
        print("Average payoff for player 2: %f " % avg_payoff_p2)
        ############################################
        
        # Plot the moves
        plt.title("Playing a matrix game")
        plt.scatter(n,m1+0.05, color = "red")
        plt.scatter(n,m2-0.05, color = "blue")
        plt.show()
        plt.pause(0.1)

        
        
        
        