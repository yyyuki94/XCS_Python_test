from xcs.xcs import XCS
from xcs.environment import MuxProblemEnvironment

if __name__ == '__main__':
    mux = MuxProblemEnvironment(2, max_iter=1000)
    xcs = XCS(mux, N=50, theta_mna=2)
    xcs.run_experiment()