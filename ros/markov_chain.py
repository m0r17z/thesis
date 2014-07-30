import numpy as np
import cPickle

class Markov_Chain:

    def __init__(self):
        self.transition_table = np.zeros((13,13))
        self.transition_table = cPickle.load(open('./posterior_table.pkl', 'rb'))

    def lookup(self, old_state, new_state):
        return self.transition_table[old_state][new_state]

class Markov_Chain_2nd:

    def __init__(self):
        self.transition_table = np.zeros((13,13,13))
        self.transition_table = cPickle.load(open('./posterior_table_2nd.pkl', 'rb'))

    def lookup(self, old_state, new_state):
        return self.transition_table[old_state][new_state]