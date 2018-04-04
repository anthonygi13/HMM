
class HMM():
    """ Define an HMM"""

    def __init__(self, letters_number, states_number, initial, transitions, emissions):
        # The number of letters
        self.__letters_number = letters_number
        # The number of states
        self.__states_number = states_number
        # The vector defining the initial weights
        self.__initial = initial
        # The array defining the transitions
        self.__transitions = transitions
        # The list of vectors defining the emissions
        self.__emissions = emissions

    @property
    def letters_number(self):
        return self.__letters_number

    @property
    def states_number(self):
        return self.__states_number

    @property
    def initial(self):
        return self.__initial

    @property
    def transitions(self):
        return self.__transitions

    @property
    def emissions(self):
        return self.__emissions
    


    def load(self, adr):
        """charge l'adresse"""

        data = open(adr, 'r')
        line = data.readline()

        for char in line:
            if char == '#':
                data.readline()
                self.nbL =

            if column_number == n_c1:
                if char != " " and char != column_separator and char != "\n":
                    c1 += char
            if column_number == n_c2:
                if char != " " and char != column_separator and char != "\n":
                    c2 += char
            if column_number > max([n_c1, n_c2]):
                break
        if c1 == "":
            c1 = None
        if c2 == "":
            c2 = None
        for i in range(nb):
            if i == 2:
                d = l.split()
                s1 = int(d[0])  # sommet 1

            l = f.readline()

        f.close()