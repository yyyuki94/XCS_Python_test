import numpy as np

from xcs.classifier import Classifier


class Population:
    def __init__(self, N: int, L: int, n_act: int, theta_del, delta, empty=True):
        self.N = N
        self.L = L
        self.n_act = n_act
        self.theta_del = theta_del
        self.delta = delta
        self.clf_list = [] if empty else [Classifier(L, n_act, 0, random=True) for _ in range(N)]

    def __iter__(self):
        self.__idx_current = 0
        return self

    def __next__(self):
        if self.__idx_current == len(self):
            raise StopIteration()

        idx = self.__idx_current
        self.__idx_current += 1

        return self.clf_list[idx]

    def __getitem__(self, idx):
        return self.clf_list[idx]

    def __len__(self):
        return len(self.clf_list)

    def get_list_of_clfattr(self, key):
        tmp = []
        for i in range(len(self)):
            tmp.append(self[i][key])
        return np.array(tmp)

    def append(self, clf):
        self.clf_list.append(clf)

    def remove(self, clf):
        for c in self.clf_list:
            if (c["condition"] == clf["condition"]).all():
                self.clf_list.remove(c)

    def print(self):
        for c in self:
            c.print()

    def delete_from_population(self):
        def deletion_vote(cl, ave_fitness):
            vote = cl["act_size"] * cl["numerosity"]
            if (cl["experience"] > self.theta_del) and \
                    (cl["fitness"] / cl["numerosity"] < self.delta * ave_fitness):
                vote = vote * ave_fitness / (cl["fitness"] / cl["numerosity"])

            return vote

        if len(self) <= self.N:
            return

        ave_fitness = np.sum(self.get_list_of_clfattr("fitness")) / np.sum(self.get_list_of_clfattr("numerosity"))

        vote_sum = 0
        for cl in self:
            vote_sum = vote_sum + deletion_vote(cl, ave_fitness)

        choice_point = np.random.rand() * vote_sum
        vote_sum = 0
        for cl in self:
            vote_sum = vote_sum + deletion_vote(cl, ave_fitness)
            if vote_sum > choice_point:
                if cl["numerosity"] > 1:
                    cl["numerosity"] -= 1
                else:
                    del self.clf_list[self.__idx_current - 1]

                return
