


import pandas as pd
import matplotlib.pyplot as plt

#cpu_data = pd.read_csv("sysinfo.csv")
# print(cpu_data.head())


import sys
cpu_data = pd.read_csv(sys.argv[1])

import numpy as np




plt.figure(figsize=(12, 4))
# plt.plot(cpu_data['timestamp'], cpu_data['temperature'])
plt.plot(cpu_data['timestamp'], cpu_data['temperature'], 'b.', alpha=0.5, label='Measured') #making everything blue


# plt.legend()
# plt.show()


import numpy as np


from statsmodels.nonparametric.smoothers_lowess import lowess 
# source: https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html



# LOESS smoothing
loess_smoothed = lowess(cpu_data['temperature'], cpu_data.index, frac=0.04) #frac=fraction of data used for each local regression, 0.04 = 4% of the data is used for each local fit
plt.plot(cpu_data['timestamp'], loess_smoothed[:, 1], 'r-', label='LOESS')
# plt.legend()
# plt.show()


# kalman smoothing
from pykalman import KalmanFilter


kalman_data = cpu_data[['temperature', 'cpu_percent', 'sys_load_1', 'fan_rpm']]

initial_state = kalman_data.iloc[0]
observation_covariance = np.diag([2, 10, 1, 50]) ** 2
transition_covariance = np.diag([0.5, 1, 0.5, 5]) ** 2

transition = [
    [0.99, 0.5,  0.2,  -0.001],
    [0.1,  0.4,  2.1,   0.0  ],
    [0.0,  0.0,  0.95,  0.0  ],
    [0.0,  0.0,  0.0,   1.0  ],
]

kf = KalmanFilter(
    initial_state_mean=initial_state,
    initial_state_covariance=observation_covariance,
    observation_covariance=observation_covariance,
    transition_covariance=transition_covariance,
    transition_matrices=transition,
)
kalman_smoothed, _ = kf.smooth(kalman_data)
plt.plot(cpu_data['timestamp'], kalman_smoothed[:, 0], 'g-', label='Kalman')

plt.legend()
plt.show()


plt.savefig('cpu.svg')