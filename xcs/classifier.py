import numpy as np


class Classifier:
    p_I = np.finfo(np.float32).eps
    e_I = np.finfo(np.float32).eps
    f_I = np.finfo(np.float32).eps

    def __init__(self, L: int, time: int, random=False):
        self.condition = np.zeros(L, dtype=np.uint8) if not random else np.random.randint(0, 3, L, dtype=np.uint8)
        self.action = 0
        self.prediction = self.p_I
        self.error = self.e_I
        self.fitness = self.f_I
        self.experience = 0
        self.time_stamp = time
        self.act_size = 1
        self.numerosity = 1

    def initialize(self):
        self.condition = np.zeros(len(self.condition), dtype=np.uint8)
        self.action = 0

    def __str__(self):
        c = "".join(map(str, self.condition))
        c = c.replace("2", "#")
        a = str(self.action)
        p = f"{self.prediction:.2f}"
        f = f"{self.fitness:.2f}"
        e = f"{self.error:.2f}"
        exp = f"{self.experience:d}"
        n = f"{self.numerosity:d}"
        outputs = [c, a, p, e, f, exp, n]

        return ",".join(outputs)

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def print(self):
        c = "".join(map(str, self.condition))
        c = c.replace("2", "#")
        a = str(self.action)
        p = f"{self.prediction:10.2f}"
        f = f"{self.fitness:6.2f}"
        e = f"{self.error:6.2f}"
        exp = f"{self.experience:05d}"
        n = f"{self.numerosity:05d}"
        out = [c, ":", a, "=>", p, "　　", "e=", e, ", F=", f, ', Exp=', exp, ', N=', n]
        out = "".join(out)
        print(out)
