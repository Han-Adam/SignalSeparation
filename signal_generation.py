from scipy.signal import stft, istft
import numpy as np
import matplotlib.pyplot as plt

# 振动信号参数设置
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
# 短时傅里叶变换参数设置
window= 'hann'
nperseg = 256
noverlap = 252
nfft = 1023
n = 63*(nperseg-noverlap)+ nperseg
division=15
divide= int(2*N/n)-1 #6


# 依次运行step1，2，3的内容，可以得到分割好的数据集


# step 1

# fault = []
# for i in range(K):
#     new= y0*np.exp(-g*2*np.pi*fn*t0)*np.sin(2*np.pi*fn*((1-g**2)**0.5)*t0)
#     bear= np.concatenate([fault,new])

# alpha= 0.8
# container= np.empty(shape=[division,division,division,3,N])
# for i in range(division):
#     for j in range(division):
#         for k in range(division):
#             theta_1=2*np.pi*i/division
#             theta_2=2*np.pi*j/division
#             theta_3=2*np.pi*k/division

#             A_1= alpha*np.cos(2*np.pi*fr*t+theta_1) + 1
#             A_2= alpha*np.cos(2*np.pi*fr*t+theta_2) + 1
#             gear=2*np.sin(2*np.pi*(170*t+beta*np.sin(2*np.pi*10*t))+theta_3)

#             container[i,j,k,0]=A_1*bear
#             container[i,j,k,1]=A_2*gear
#             container[i,j,k,2]=container[i,j,k,0]+container[i,j,k,1]+0.3*np.random.randn(N)

#             print(i,j,k)

# np.save('./Data/raw_signal.npy',container)

#step 2
# container=np.load('./Data/raw_signal.npy')
# print(container.shape)

# divide= int(2*N/n)-1 #6

# bear_signal=np.empty([division**3*divide,n])
# gear_signal=np.empty([division**3*divide,n])
# mix_signal= np.empty([division**3*divide,n])
# for i in range(division):
#     for j in range(division):
#         for k in range(division):
#             for l in range(divide):
#                 index1=((i*division+j)*division+k)*divide+l
#                 index2=int(l*n/2)
#                 bear_signal[index1]=container[i,j,k,0][index2:index2+n]
#                 gear_signal[index1]=container[i,j,k,1][index2:index2+n]
#                 mix_signal[index1]=container[i,j,k,2][index2:index2+n]

#                 print(i,j,k,l)

# np.save('./Data/bear_signal.npy',bear_signal)
# np.save('./Data/gear_signal.npy',gear_signal)
# np.save('./Data/mix_signal.npy',mix_signal)


# 建议在运行step3的时候，可以把数据集的文件夹建在Signal Seperation外面
# 如果放在里面，每次打开Pycharm都要扫描好久。
# step 3
# bear_signal=np.load('./Data/bear_signal.npy')
# gear_signal=np.load('./Data/gear_signal.npy')
# mix_signal=np.load('./Data/mix_signal.npy')
# length=fault_signal.shape[0]
# plt.plot(mix_signal[5])
#
# plt.show()

# bear_magnitude=np.empty(shape=[length,512,128])
# gear_magnitude=np.empty(shape=[length,512,128])
# mix_magnitude=np.empty(shape=[length,512,128])
# mix_phase=np.empty(shape=[length,512,128])
# for i in range(length):
#     f, t, STFT= stft(x=bear_signal[i],
#                         fs=fs,
#                         window= window,
#                         nperseg = nperseg,
#                         noverlap = noverlap,
#                         nfft = nfft,
#                         boundary='zeros',
#                         padded = True)
#     bear_magnitude=np.abs(STFT)
#     np.save('./Data/bear_magnitude/data'+str(i)+'.npy',fault_magnitude)

#     f, t, STFT= stft(x=gear_signal[i],
#                         fs=fs,
#                         window= window,
#                         nperseg = nperseg,
#                         noverlap = noverlap,
#                         nfft = nfft,
#                         boundary='zeros',
#                         padded = True)
#     gear_magnitude=np.abs(STFT)
#     np.save('./Data/gear_magnitude/data'+str(i)+'.npy',gear_magnitude)

#     f, t, STFT= stft(x=mix_signal[i],
#                         fs=fs,
#                         window= window,
#                         nperseg = nperseg,
#                         noverlap = noverlap,
#                         nfft = nfft,
#                         boundary='zeros',
#                         padded = True)
#     mix_magnitude=np.abs(STFT)
#     mix_phase=np.angle(STFT)
#     np.save('./Data/mix_magnitude/data'+str(i)+'.npy',mix_magnitude)
#     np.save('./Data/mix_phase/data'+str(i)+'.npy',mix_phase)

#     print(i)



