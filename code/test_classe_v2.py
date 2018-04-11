# Mac Donald  Kilian
# test_classe_v2
# crée le 10/04/2018

from classe import *
import numpy as np

import unittest

class TestHMM_2(unittest.TestCase):
    '''classe de tests de la classe HMM'''

    def setUp(self):
        self.hmm1 = HMM.load('test1.txt') # lui pas de probleme
        self.hmm2 = HMM.load('test2.txt') # pas de probleme
        #self.hmm3 = HMM.load('test3.txt')  # attention , lui somme != 0

    def test_Error1(self):
        self.assertRaises(AssertionError, HMM, 'test3.txt.txt')

    def test_load(self):
        self.assertTrue(self.hmm1.letters_number == 2)
        self.assertTrue(self.hmm1.states_number == 2)
        self.assertTrue(self.hmm1.initial.all == np.array([(0.5 , 0.5)]).all)                  #essayer individuellement, voir en bas
        self.assertTrue(self.hmm1.transitions.all == np.array([(0.9, 0.1), (0.1, 0,9)]).all)
        self.assertTrue(self.hmm1.emissions.all == np.array([(0.5, 0.5), (0.7, 0, 3)]).all)

    def test_load1(self):
        self.assertTrue(self.hmm2.letters_number == 2)
        self.assertTrue(self.hmm2.states_number == 3)
        self.assertTrue(self.hmm2.initial.all == np.array([0.5 , 0.2 , 0.3]).all)
        self.assertTrue(self.hmm2.transitions.all == np.array([ (0.8,  0.1 , 0.1), (0.8 , 0.9 , 0.1),  (0.8 , 0.9 , 0.1)]).all)
        self.assertTrue(self.hmm2.emissions.all == np.array([ (0.5 , 0.5),  (0.6 , 0.4 ), (0.9 ,  0.1)]).all)


if __name__ == "__main__":
    unittest.main()



'''
    def testError2(self):
        n = 1
        L = [[2], [0, 0], [0, 1], [0, -1], [1, 2]]
        self.assertRaises(AssertionError, IneqSys, n, L)

    def testSystemeSimple1(self):
        t = self.i1.deciderSystemeSimple()
        self.assertTrue(t == False)

    def testSystemeSimple2(self):
        t = self.i2.deciderSystemeSimple()
        self.assertTrue(t == True)

    def testElimFM1(self):
        self.assertFalse(self.i1.elimFM())

    def testElimFM2(self):
        self.assertTrue(self.i2.elimFM())

    def testElimFM3(self):
        self.assertTrue(self.i3.elimFM())

    def testElimFM4(self):
        self.assertFalse(self.i4.elimFM())

    def testElimFM5(self):
        self.assertTrue(self.i5.elimFM())

    def testElimFM6(self):
        self.assertTrue(self.i6.elimFM())
'''

print ((np.array([1, 2, 3]) == np.array([1, 2, 3])).all())