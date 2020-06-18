from xcs.xcs import XCS
from xcs.environment import MuxProblemEnvironment

if __name__ == '__main__':
    mux = MuxProblemEnvironment(2, max_iter=10000)
    xcs = XCS(mux, N=50, theta_mna=2, eps_0=10,
              do_ga_subsumption=True, do_actionset_subsumption=True)
    xcs.run_experiment()
