###########################
#  Fichier de test        #
# 05/04/18                #
# la communauté de l'info #
###########################

from classe import *

import unittest

class TestHMM(unittest.TestCase):
    '''classe de tests de la classe HMM'''

    def setUp(self):
        self.hmm1 = HMM('test1.txt') # lui pas de probleme
        self.hmm2 = HMM('test2.txt') # pas de probleme
        self.hmm3 = HMM('test3.txt')  # attention , lui somme != 0

    def test_Error1(self):
        self.assertTrue(self.hmm2.load())

    def test_Error2(self):
        self.assertTrue(self.hmm3.load())



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