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
        self.A = HMM(2, 2, np.array([0.5, 0.5]), np.array([[0.9, 0.1], [0.1, 0.9]]),np.array([[0.5, 0.5], [0.7, 0.3]]))
        self.B = HMM(2, 2, np.array([0.741, 0.259]), np.array([[0.0115, 0.9885], [0.5084, 0.4916]]),
                         np.array([[0.4547, 0.5453], [0.2089, 0.7911]]))

    def test_HMM(self):
        self.assertRaises(ValueError, HMM, 0, 2, np.array([0.5, 0.5]), np.array([[0.9, 0.1], [0.1, 0.9]]),
                          np.array([[0.5, 0.5], [0.7, 0.3]]))
        self.assertRaises(ValueError, HMM, 2, 0, np.array([0.5, 0.5]), np.array([[0.9, 0.1], [0.1, 0.9]]),
                          np.array([[0.5, 0.5], [0.7, 0.3]]))
        self.assertRaises(TypeError, HMM, 2, 2, [0.5, 0.5], np.array([[0.9, 0.1], [0.1, 0.9]]),
                          np.array([[0.5, 0.5], [0.7, 0.3]]))
        self.assertRaises(ValueError, HMM, 2, 2, np.array([-0.5, 0.5]), np.array([[0.9, 0.1], [0.1, 0.9]]),
                          np.array([[0.5, 0.5], [0.7, 0.3]]))
        self.assertRaises(ValueError, HMM, 2, 2, np.array([0.5, 1.5]), np.array([[0.9, 0.1], [0.1, 0.9]]),
                          np.array([[0.5, 0.5], [0.7, 0.3]]))
        self.assertRaises(ValueError, HMM, 2, 2, np.array([0.5, 0.5]), np.array([[0.9, 0.1], [0.1, 0.8]]),
                          np.array([[0.5, 0.5], [0.7, 0.3]]))
        self.assertRaises(ValueError, HMM, 2, 2, np.array([0.5, 0.5]), np.array([[0.9, 0.1], [0.1, 0.9]]),
                          np.array([[0.5, 0.5], [0.75, 0.3]]))


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

    def test_save_load(self):
        h = self.A
        h.save("./temp")
        h = HMM.load("./temp")
        self.assertEqual(h.letters_number, 2)
        self.assertEqual(h.states_number, 2)
        np.testing.assert_array_equal(h.initial, np.array([0.5, 0.5]))
        np.testing.assert_array_equal(h.transitions, np.array([[0.9, 0.1], [0.1, 0.9]]))
        np.testing.assert_array_equal(h.emissions, np.array([[0.5, 0.5], [0.7, 0.3]]))


    def test_save(self):
        self.hmm1.save('test_de_test1.txt')
        self.assertTrue(self.hmm1.letters_number == HMM.load("test_de_test1.txt").letters_number)
        self.assertTrue(self.hmm1.states_number == HMM.load("test_de_test1.txt").states_number)
        self.assertTrue((self.hmm1.initial == HMM.load("test_de_test1.txt").initial).all())
        self.assertTrue((self.hmm1.transitions == HMM.load("test_de_test1.txt").transitions).all())
        self.assertTrue((self.hmm1.emissions == HMM.load("test_de_test1.txt").emissions).all())

    def test_generate_random(self):
        sequence = self.hmm1.generate_random(5)
        self.assertTrue(type(sequence) == tuple)
        self.assertTrue(len(sequence) == 5)

    def test_pfw(self):
        self.assertTrue(self.hmm1.pfw([0]) == 0.6)
        self.assertTrue(self.hmm1.pfw([1]) == 0.4)
        self.assertTrue(self.hmm1.pfw([0,0]) == 0.368)


    def test_pbw(self):
        self.assertTrue(self.hmm1.pbw([0]) == 0.6)
        self.assertTrue(self.hmm1.pbw([1]) == 0.4)
        self.assertTrue(self.hmm1.pbw([0,0]) == 0.368)

    def test_PFw_PBw(self):
        h = self.A
        self.assertEqual(h.pfw([0]), 0.6)
        self.assertEqual(h.pfw([1]), 0.4)
        for i in range(100):
            w = h.generate_random(10)
            self.assertAlmostEqual(h.pfw(w), h.pbw(w))


    def test_predit(self):
        for i in range(100):
            h = HMM.gen_HMM(2, 5)
            w = h.generate_random(10)
            w0 = w + (0,)
            w1 = w + (1,)
            x = h.predit(w)
            if h.pfw(w0) > h.pfw(w1):
                self.assertEqual(0, x)
            else:
                self.assertEqual(1, x)

    def test_list_len1(self):
        L = list_rand_sum_2_dim(2,5)
        sum = 0
        for i in range(5):
            sum += float(L[0, i])
        self.assertTrue(np.isclose([sum], [1]))


        sum = 0
        for i in range(5):
            sum += float(L[1, i])
        self.assertTrue(np.isclose([sum], [1]))

    def test_list_len2(self):
        L = list_rand_sum_2_dim(3,5)
        sum = 0
        for i in range(5):
            sum += float(L[0, i])
        self.assertTrue(np.isclose([sum], [1]))

        sum = 0
        for i in range(5):
            sum += float(L[1, i])
        self.assertTrue(np.isclose([sum], [1]))

        sum = 0
        for i in range(5):
            sum += float(L[2, i])
        self.assertTrue(np.isclose([sum], [1]))


    def test_Viterbi(self):
        h = self.B
        w = (1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1)
        (lc, p) = h.viterbi(w)
        self.assertEqual(lc, (0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1))
        self.assertAlmostEqual(p, -15.816435284201352)


    def test_BaumWelch(self):
        h = self.A
        w = (0, 1)
        h.bw1([w])
        np.testing.assert_allclose(h.initial, np.array([0.51724138, 0.48275862]))
        np.testing.assert_allclose(h.transitions, np.array([[0.9375, 0.0625], [0.15625, 0.84375]]))
        np.testing.assert_allclose(h.emissions, np.array([[0.48, 0.52], [0.52336449, 0.47663551]]))


    def tearDown(self):
        self.A = None
        self.B = None




if __name__ == "__main__":
    unittest.main()

