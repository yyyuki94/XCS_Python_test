import numpy as np

from xcs.matchset import MatchSet


class ActionSet:
    def __init__(self, match_set: MatchSet, act: int):
        self.A = []
        for cl in match_set:
            if cl["action"] == act:
                self.A.append(cl)

    def __iter__(self):
        self.__idx_current = 0
        return self

    def __next__(self):
        if self.__idx_current == len(self):
            raise StopIteration()

        idx = self.__idx_current
        self.__idx_current += 1

        return self.A[idx]

    def __getitem__(self, idx):
        return self.A[idx]

    def __len__(self):
        return len(self.A)

    def remove(self, clf):
        for c in self.A:
            if (c["condition"] == clf["condition"]).all():
                self.A.remove(c)


class PredictArray:
    def __init__(self, match_set: MatchSet):
        self.act_all = np.arange(match_set.act_min, match_set.act_max+1)
        self.PA = np.zeros(self.act_all.shape[0])
        self.PA[:] = np.nan
        self.FSA = np.zeros(self.act_all.shape[0])

        for cl in match_set:
            act_idx = self.__get_matched_idx(cl["action"])
            if np.isnan(self.PA[act_idx]):
                self.PA[act_idx] = cl["prediction"] * cl["fitness"]
            else:
                self.PA[act_idx] += cl["prediction"] * cl["fitness"]
            self.FSA[act_idx] += cl["fitness"]

        for act in range(len(self.act_all)):
            if self.FSA[act] != 0:
                self.PA[act] = self.PA[act] / self.FSA[act]

    def __iter__(self):
        self.__idx_current = 0
        return self

    def __next__(self):
        if self.__idx_current == len(self):
            raise StopIteration()

        idx = self.__idx_current
        self.__idx_current += 1

        return self.PA[idx]

    def __getitem__(self, idx):
        return self.PA[idx]

    def __len__(self):
        return len(self.PA)

    def __get_matched_idx(self, act):
        idx = np.arange(len(self.act_all))[self.act_all == act]
        return idx

    def select_action(self, p_explr):
        if np.random.rand() < p_explr:
            idx = np.random.choice(np.arange(len(self.PA))[np.logical_not(np.isnan(self.PA))])
            return self.act_all[idx]
        else:
            return self.act_all[np.nanargmax(self.PA)]
