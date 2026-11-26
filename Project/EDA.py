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

# =============Load  data==================
data = '/Users/gustafutterstrom/Desktop/Utbildning/Progg/ClonedRepos/TimeSeriesAnalysis_Project/Project/projectData26.mat'
mat_data = scipy.io.loadmat(data)
nbdData = mat_data['nbdData'].flatten()
tempData = mat_data['tempData'].flatten()
waterflowData = mat_data['waterflowData'].flatten()

"""""
print("Keys in .mat file:", mat_data.keys())
print(len(nbdData))
print(len(tempData))
print(len(waterflowData))
print(f"Data shape: {nbdData.shape}")
"""""

# Plot the data
def plot_data(data, title="", xlable="",ylable=""):
    plt.figure(figsize=(12, 4))
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlable)
    plt.ylabel(ylable)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def my_filter(A, C, y):
    
    e_hat = signal.lfilter(A, C, y)
    na=len(A)-1
    return e_hat[na:]

##===== Data-split=============
train_size = int(0.65*len(waterflowData))
val_size=int(0.20*len(waterflowData))
train_wf = waterflowData[:train_size]
val_wf =waterflowData[train_size:train_size+val_size]
test_wf=waterflowData[train_size+val_size:]
print(len(train_wf))
print(len(val_wf))
print(len(test_wf))

##===== Transform-split=============

#lambda_max, offsetValue = box_cox(waterflowData, plotIt=True, transform=False)
#print(lambda_max)
#plot_data(train_wf)
transformed_wf= np.log(train_wf)
#plot_data(transformed_wf)

test_mean(transformed_wf)
transformed_wf-=np.mean(transformed_wf)
test_mean(transformed_wf)

plotACFnPACF(transformed_wf, noLags=50)

model_s = estimateARMA(transformed_wf, A=1, C=0, plot=False)
model_s.summary()

# Compute residuals
e_hat_s = my_filter(model_s.A, model_s.C, transformed_wf)

# Analyze residuals
Sacf_estS,_=plotACFnPACF(e_hat_s, noLags=30, signLvl=0.05,return_val=True)
normplot(e_hat_s)
check_if_normal(Sacf_estS[1:], which_test='J')
whiteness_test(e_hat_s, alpha=0.05)
print("Residual var:",np.var(e_hat_s))
plt.plot(e_hat_s, label="Estimated residuals", alpha=0.7)
plt.title(f"ARMA({2},{0}): estimated residuals")
plt.xlabel("months")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()