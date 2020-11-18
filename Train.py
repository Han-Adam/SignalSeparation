import numpy as np
import keras

from keras.utils import np_utils
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.optimizers import adam

import copy as copy

F=512
T=128

train_num= 15000
batch= 10
step= 300
epoch= 2

with open('./Model/Untrained_Model.json', 'r') as file:
    model_json = file.read()
new_model = model_from_json(model_json)
new_model.compile(loss='mean_squared_error', optimizer="adam")

# 我只训练了一个1个epoch，在外面套for就能训练多个epoch
for i in range(int(train_num/step)):
    # 图片太大，我就用这种方式一张一张的读进来的
    train_feed=np.empty(shape=[step,F,T])
    train_target=np.empty(shape=[step,F,T])
    for j in range(step):
        train_feed[j]=np.load('./Data/mix_magnitude/data'+str(step*i+j)+'.npy')
        # 如果要分离gear，就load gear，如果要分离bear，就load bear
        train_target[j]=np.load('./Data/gear_magnitude/data'+str(step*i+j)+'.npy')
    train_feed=train_feed.reshape([step,F,T,1])
    train_target=np.reshape(train_target,[step,F,T,1])

    new_model.fit(train_feed, train_target, epochs=epoch, batch_size=batch)
    print(i)

new_model.save('./Trained_Model_2.json.h5')
