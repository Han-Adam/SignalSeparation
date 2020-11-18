import numpy as np
import keras

from keras.utils import np_utils
from keras.optimizers import Adam
from keras.models import model_from_json,load_model
from keras.optimizers import adam
from scipy.signal import stft, istft
import copy as copy

fs = 1024
window= 'hann'
nperseg = 256
noverlap = 252
nfft = 1023
n = 63*(nperseg-noverlap)+ nperseg

F=512
T=128

feed= np.empty(shape=[2,F,T])
phase= np.empty(shape=[2,F,T])
for i in range(2):
    feed[i]=np.load('./Data/mix_magnitude/data'+str(20000+i)+'.npy')
    phase[i]=np.load('./Data/mix_phase/data'+str(20000+i)+'.npy')
feed= np.reshape(feed,[2,F,T,1])
# 这里的load_model和我们拿到的数据类型要匹配
Denoising_Model = load_model('./Model/Trained_Model_2.json.h5')
Denoised_Data = Denoising_Model.predict(feed)
Denoised_Data = np.reshape(Denoised_Data,[2,F,T])
# u-net recover的幅度谱，加上原始信号的相位谱，还原真正的时频谱。
recovered_spectrum = Denoised_Data*(np.cos(phase)+1j*np.sin(phase))

# recovered_spectrum = np.reshape(recovered_spectrum, shape=[374,512,1023,1])
# recovered_signal = np.empty(shape=[374,n])
recovered_signal= np.empty(shape=[2,n])
for i in range(2):
    # 逆短时傅里叶变换
    t2,signal= istft(Zxx=recovered_spectrum[i],
                                        fs=fs,
                                        window= window,
                                        nperseg = nperseg,
                                        noverlap = noverlap,
                                        nfft = nfft,
                                        boundary='zeros')
    recovered_signal[i]=signal

np.save('./Data/recover_spectrum_2.npy',recovered_signal)