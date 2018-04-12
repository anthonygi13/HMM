###########################
#  Fichier de test        #
# 05/04/18                #
# la communauté de l'info #
###########################

from classe import *
import numpy as np

import unittest

class TestHMM(unittest.TestCase):
    '''classe de tests de la classe HMM'''

    def setUp(self):
        self.hmm1 = HMM.load('test1.txt') # lui pas de probleme
        self.hmm2 = HMM.load('test2.txt') # pas de probleme

    def test_Error1(self):
        '''avec une somme >1 pour une ligne'''
        self.assertRaises(ValueError, HMM, 2, 3, np.array([0.5, 0.2, 0.3]), np.array([(0.8, 0.1, 0.3), (0.8, 0.9, 0.3), (0.8, 0.9, 0.3)]), np.array([(0.3, 0.3), (0.6, 0.3), (0.5, 0.5)]))

    def test_Error2(self):
        '''avec une valeur negative'''
        self.assertRaises(ValueError, HMM, 2, 3, np.array([[ 0.5 , 0.2 , -0.3]]), np.array([[ 0.8,  0.1 , 0.1], [ 0.8 , 0.1 , 0.1], [ 0.7 , 0.2 , 0.1]]), np.array([[ 0.5 , 0.5 ], [ 0.6 , 0.4 ], [ 0.9 ,  0.1]]))

    def test_load(self):
        self.assertTrue(self.hmm1.letters_number == 2)
        self.assertTrue(self.hmm1.states_number == 2)
        self.assertTrue((self.hmm1.initial == np.array([[0.5 , 0.5]])).all)
        self.assertTrue((self.hmm1.transitions == np.array([[0.9, 0.1], [0.1, 0.9]])).all())
        self.assertTrue((self.hmm1.emissions == np.array([[0.5, 0.5], [0.7, 0.3]])).all())

    def test_load1(self):
        self.assertTrue(self.hmm2.letters_number == 2)
        self.assertTrue(self.hmm2.states_number == 3)
        self.assertTrue((self.hmm2.initial == np.array([[ 0.5 , 0.2 , 0.3]])).all())
        self.assertTrue((self.hmm2.transitions == np.array([[ 0.8,  0.1 , 0.1], [ 0.8 , 0.1 , 0.1], [ 0.7 , 0.2 , 0.1]])).all())
        self.assertTrue((self.hmm2.emissions == np.array([[ 0.5 , 0.5 ], [ 0.6 , 0.4 ], [ 0.9 ,  0.1]])).all())

    def test_generate_random(self):
        sequence = self.hmm1.generate_random(5)
        self.assertTrue(type(sequence) == np.ndarray)
        self.assertTrue(sequence.shape[0] == 5)

    def test_save(self):
        self.hmm1.save('test_de_test1.txt')
        self.assertTrue(self.hmm1.letters_number == HMM.load("test_de_test1.txt").letters_number)
        self.assertTrue(self.hmm1.states_number == HMM.load("test_de_test1.txt").states_number)
        self.assertTrue((self.hmm1.initial == HMM.load("test_de_test1.txt").initial).all())
        self.assertTrue((self.hmm1.transitions == HMM.load("test_de_test1.txt").transitions).all())
        self.assertTrue((self.hmm1.emissions == HMM.load("test_de_test1.txt").emissions).all())


if __name__ == "__main__":
    unittest.main()

