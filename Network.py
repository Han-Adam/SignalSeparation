import keras
from keras.layers import (BatchNormalization,
    Input,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    ELU,
    LeakyReLU,
    Multiply,
    ReLU,
    Softmax)
from keras.models import Model, model_from_json

filter=[16, 32, 64, 128, 256, 512]
kernel_size=[5,5]
strides=[2,2]
padding='same'
conv_tran_act='relu'

input = Input(shape=[512, 128,1])

# 这个文件里面，我复现了在二分割情况下，最基本的U-Net
# 原文中的U-net根据不同的情况，有好多变量，这个方便理解一点

# 1st layer
conv1 = Conv2D(filters=filter[0],
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=padding,
                            activation=None)(input)
batch1 = BatchNormalization(axis=-1)(conv1)
rel1 = LeakyReLU(alpha=0.2)(batch1)
 # 2nd layer
conv2 = Conv2D(filters=filter[1],
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=padding,
                            activation=None)(rel1)
batch2 = BatchNormalization(axis=-1)(conv2)
rel2 = LeakyReLU(alpha=0.2)(batch2)
# 3rd layer
conv3 = Conv2D(filters=filter[2],
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=padding,
                            activation=None)(rel2)
batch3 = BatchNormalization(axis=-1)(conv3)
rel3 = LeakyReLU(alpha=0.2)(batch3)
# 4th layer
conv4 = Conv2D(filters=filter[3],
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=padding,
                            activation=None)(rel3)
batch4 = BatchNormalization(axis=-1)(conv4)
rel4 = LeakyReLU(alpha=0.2)(batch4)

 # 5th layer
# conv5 = Conv2D(filters=filter[4],
#                             kernel_size=kernel_size,
#                             strides=strides,
#                             padding=padding,
#                             activation=None)(rel4)
# batch5 = BatchNormalization(axis=-1)(conv5)
# rel5 = LeakyReLU(alpha=0.2)(batch5)

# 6th layer
conv6 = Conv2D(filters=filter[5],
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=padding,
                            activation=None)(rel4)#(rel5)

# 7th layer
# up5 = Conv2DTranspose(filters=filter[4],
#                                         kernel_size=kernel_size,
#                                         strides=strides,
#                                         padding=padding,
#                                         activation=conv_tran_act)(conv6)
# up_batch5 = BatchNormalization(axis=-1)(up5)
# drop5 = Dropout(0.5)(up_batch5)
# merge5 = Concatenate(axis=-1)([conv5, drop5])

# 8th layer
up4 = Conv2DTranspose(filters=filter[3],
                                        kernel_size=kernel_size,
                                        strides=strides,
                                        padding=padding,
                                        activation=conv_tran_act)(conv6)#(merge5)
up_batch4 = BatchNormalization(axis=-1)(up4)
drop4 = Dropout(0.5)(up_batch4)
merge4 = Concatenate(axis=-1)([conv4, drop4])
# 9th layer
up3 = Conv2DTranspose(filters=filter[2],
                                        kernel_size=kernel_size,
                                        strides=strides,
                                        padding=padding,
                                        activation=conv_tran_act)(merge4)
up_batch3 = BatchNormalization(axis=-1)(up3)
drop3 = Dropout(0.5)(up_batch3)
merge3 = Concatenate(axis=-1)([conv3, drop3])
# 10th layer
up2 = Conv2DTranspose(filters=filter[1],
                                        kernel_size=kernel_size,
                                        strides=strides,
                                        padding=padding,
                                        activation=conv_tran_act)(merge3)
up_batch2 = BatchNormalization(axis=-1)(up2)
merge2 = Concatenate(axis=-1)([conv2, up_batch2])
# 11th layer
up1 = Conv2DTranspose(filters=filter[0],
                                        kernel_size=kernel_size,
                                        strides=strides,
                                        padding=padding,
                                        activation=conv_tran_act)(merge2)
up_batch1= BatchNormalization(axis=-1)(up1)
merge1 = Concatenate(axis=-1)([conv1, up_batch1])
# tail
tail = Conv2DTranspose(filters=1,
                                        kernel_size=kernel_size,
                                        strides=strides,
                                        padding=padding,
                                        activation=conv_tran_act)(merge1)
tail = BatchNormalization(axis=-1)(tail)
tail = Conv2D(1,
                        (4, 4),
                        dilation_rate=(2, 2),
                        activation='sigmoid',
                        padding='same') (tail)
output = Multiply(name='output')([tail, input])


Total_Model = Model(inputs=input, outputs=output)
Total_Model.compile(optimizer='adam', loss='mean_squared_error')

model_json = Total_Model.to_json()
with open( ".Model/Untrained_Model.json", 'w') as file:
    file.write(model_json)

