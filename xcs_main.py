from xcs.xcs import XCS
from xcs.environment import MuxProblemEnvironment
import pandas as pd

if __name__ == '__main__':
    mux = MuxProblemEnvironment(2, max_iter=10000)
    xcs = XCS(mux, N=1000, theta_mna=2, eps_0=10, p_explr=1.0,
              chi=0.8, mu=0.04, p_I=0.01, e_I=0.01, f_I=0.01,
              beta=0.2, do_ga_subsumption=True, do_actionset_subsumption=True)
    xcs.run_experiment()
    xcs.Pop.output_csv('./result.csv')
    mux.save_rewards('./log-reward.csv')
    reward = mux.log_table.reshape(-1, 1)
    sma = pd.DataFrame(reward)
    sma = sma.rolling(100).mean().fillna(0)
    sma.to_csv('./reward-ma100.csv', header=False, index=False)
