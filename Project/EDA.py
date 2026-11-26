# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sys
import os
import importlib
import scipy.io
import sys


sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "TimeSeriesAnalysis-main",
        "TimeSeriesAnalysis-main",
    )
)
# Import and reload to get the latest changes
import tsa_lth.analysis
import tsa_lth.modelling
import tsa_lth.tests
importlib.reload(tsa_lth.analysis)
importlib.reload(tsa_lth.modelling)
importlib.reload(tsa_lth.tests)


from tsa_lth.analysis import plotACFnPACF, normplot, xcorr, pzmap, kovarians, box_cox
from tsa_lth.modelling import estimateARMA
from tsa_lth.tests import whiteness_test, check_if_normal, test_mean

# Load Svedala temperature data
data = '/Users/gustafutterstrom/Desktop/Utbildning/Progg/ClonedRepos/TimeSeriesAnalysis_Project/Project/projectData26.mat'
mat_data = scipy.io.loadmat(data)
nbdData = mat_data['nbdData'].flatten()
tempData = mat_data['tempData'].flatten()
waterflowData = mat_data['waterflowData'].flatten()
print("Keys in .mat file:", mat_data.keys())

print(len(nbdData))
print(len(tempData))
print(len(waterflowData))
print(f"Data shape: {nbdData.shape}")

# Plot the data
plt.figure(figsize=(12, 4))
plt.plot(nbdData)
plt.title('Svedala Temperature Data')
plt.xlabel('Time (months)')
plt.ylabel('Temperature')
plt.grid(True)
plt.tight_layout()
plt.show()
