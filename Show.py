import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
fr=30            #内圈
fs = 10000   #采样频率
fn = 3000     #固有频率
y0 =15           # 位移常数
g = 0.1         #阻尼系数
T = 1/170     #重复周期
N = 2000     #采样点数
NT = int(fs*T)+1                        #单周期采样点数
K = int(N/NT)+1                        #重复次数
N = NT*K
n= np.linspace(0,N-1,N)
t = np.linspace(0,N-1,N)/fs         #采样时刻
t0 = np.linspace(0,NT-1,NT)/fs   #单周期采样时刻
fz= 2.5/170
beta= 2

t=t[0:508]

window= 'hann'
nperseg = 256
noverlap = 252
nfft = 1023
n = 63*(nperseg-noverlap)+ nperseg
division=15
divide= int(2*N/n)-1 #6

# 画图
recovered_signal=np.load('./Data/fault_signal.npy')[15000]
original_signal=np.load('./Data/mix_signal.npy')[15000]
target_signal=np.load('./Data/gear_signal.npy')[15000]
plt.subplot(3,1,1)
plt.plot(t,target_signal)
plt.subplot(3,1,2)
plt.plot(t,recovered_signal)
plt.subplot(3,1,3)
plt.plot(t,original_signal)
plt.show()

f, t, STFT= stft(x=original_signal,
                        fs=fs,
                        window= window,
                        nperseg = nperseg,
                        noverlap = noverlap,
                        nfft = nfft,
                        boundary='zeros',
                        padded = True)

plt.pcolormesh(t, f, np.abs(STFT))
plt.show()