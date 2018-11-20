'''
version:
TensorFlow==1.7.0
Keras==2.2.4
'''
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import Conv2D, Add, ZeroPadding2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ReLU
from keras.utils import plot_model

# DetNet59 Model Structure
# _______________________________________________________________________________
# Stage   |output size| kernel size | num_filters | num_blocks | stride | dilate
# _______________________________________________________________________________
# Stage 1 |  112x112  |     7x7     |     64      |     ---    |   2   |    1
# -------------------------------------------------------------------------------
#         |           |     3x3     |           max pool       |   2   | 
#         |           |     1x1     |     64       |           |       |
# Stage 2 |   56x56   |     3x3     |     64       |     3     |   1   |    1
#         |           |     1x2     |     256      |           |       |
# ------------------------------------------------------------------------------- 
#         |           |     1x1     |     128      |           |       |
# Stage 3 |   28x28   |     3x3     |     128      |     4     |   2   |    1
#         |           |     1x2     |     512      |           |       |
# -------------------------------------------------------------------------------
#         |           |     1x1     |     256      |           |       |
# Stage 4 |   14x14   |     3x3     |     256      |     6     |   2   |    1
#         |           |     1x2     |     1024     |           |       |
# -------------------------------------------------------------------------------
#         |           |     1x1     |     256      |           |       |
# Stage 5 |   14x14   |     3x3     |     256      |     3     |   1   |    2
#         |           |     1x2     |     256      |           |       |
# -------------------------------------------------------------------------------
#         |           |     1x1     |     256      |           |       |
# Stage 6 |   14x14   |     3x3     |     256      |     3     |   1   |    2
#         |           |     1x2     |     256      |           |       |
# -------------------------------------------------------------------------------
#         |    1x1    |    14x14    |        ave pool, 1000-d fc, softmax 
# -------------------------------------------------------------------------------

def res_block(x, filters_list, strides=1, use_bias=True, name=None):
    '''
    y = f3(f2(f1(x))) + x

    # Conv2D default arguments:
        strides=1
        padding='valid'
        data_format='channels_last'
        dilation_rate=1
        activation=None
        use_bias=True
    '''
    out = Conv2D(filters=filters_list[0], kernel_size=1, strides=1, use_bias=False, name='%s_1'%(name))(x)
    out = BatchNormalization(name='%s_1_bn'%(name))(out)
    out = ReLU(name='%s_1_relu'%(name))(out)

    out = Conv2D(filters=filters_list[1], kernel_size=3, strides=1, padding='same', use_bias=False, name='%s_2'%(name))(out)
    out = BatchNormalization(name='%s_2_bn'%(name))(out)
    out = ReLU(name='%s_2_relu'%(name))(out)

    out = Conv2D(filters=filters_list[2], kernel_size=1, strides=1, use_bias=False, name='%s_3'%(name))(out)
    out = BatchNormalization(name='%s_3_bn'%(name))(out)

    out = Add(name='%s_add'%(name))([x, out])
    out = ReLU(name='%s_relu'%(name))(out)    
    return out

def res_block_proj(x, filters_list, strides=2, use_bias=True, name=None):
    '''
    y = f3(f2(f1(x))) + proj(x)
    '''
    out = Conv2D(filters=filters_list[0], kernel_size=1, strides=strides, use_bias=False, name='%s_1'%(name))(x)
    out = BatchNormalization(name='%s_1_bn'%(name))(out)
    out = ReLU(name='%s_1_relu'%(name))(out)

    out = Conv2D(filters=filters_list[1], kernel_size=3, strides=1, padding='same', use_bias=False, name='%s_2'%(name))(out)
    out = BatchNormalization(name='%s_2_bn'%(name))(out)
    out = ReLU(name='%s_2_relu'%(name))(out)

    out = Conv2D(filters=filters_list[2], kernel_size=1, strides=1, use_bias=False, name='%s_3'%(name))(out)
    out = BatchNormalization(name='%s_3_bn'%(name))(out)

    x = Conv2D(filters=filters_list[2], kernel_size=1, strides=strides, use_bias=False, name='%s_proj'%(name))(x)
    x = BatchNormalization(name='%s_proj_bn'%(name))(x)

    out = Add(name='%s_add'%(name))([x, out])
    out = ReLU(name='%s_relu'%(name))(out)    
    return out

def dilated_res_block(x, filters_list, strides=1, use_bias=True, name=None):
    '''
    y = f3(f2(f1(x))) + x
    '''
    out = Conv2D(filters=filters_list[0], kernel_size=1, strides=1, use_bias=False, name='%s_1'%(name))(x)
    out = BatchNormalization(name='%s_1_bn'%(name))(out)
    out = ReLU(name='%s_1_relu'%(name))(out)

    out = Conv2D(filters=filters_list[1], kernel_size=3, strides=1, padding='same', dilation_rate=2, use_bias=False, name='%s_2'%(name))(out)
    out = BatchNormalization(name='%s_2_bn'%(name))(out)
    out = ReLU(name='%s_2_relu'%(name))(out)

    out = Conv2D(filters=filters_list[2], kernel_size=1, strides=1, use_bias=False, name='%s_3'%(name))(out)
    out = BatchNormalization(name='%s_3_bn'%(name))(out)

    out = Add(name='%s_add'%(name))([x, out])
    out = ReLU(name='%s_relu'%(name))(out)    
    return out

def dilated_res_block_proj(x, filters_list, strides=1, use_bias=True, name=None):
    '''
    y = f3(f2(f1(x))) + proj(x)
    '''
    out = Conv2D(filters=filters_list[0], kernel_size=1, strides=1, use_bias=False, name='%s_1'%(name))(x)
    out = BatchNormalization(name='%s_1_bn'%(name))(out)
    out = ReLU(name='%s_1_relu'%(name))(out)

    out = Conv2D(filters=filters_list[1], kernel_size=3, strides=1, padding='same', dilation_rate=2, use_bias=False, name='%s_2'%(name))(out)
    out = BatchNormalization(name='%s_2_bn'%(name))(out)
    out = ReLU(name='%s_2_relu'%(name))(out)

    out = Conv2D(filters=filters_list[2], kernel_size=1, strides=1, use_bias=False, name='%s_3'%(name))(out)
    out = BatchNormalization(name='%s_3_bn'%(name))(out)

    x = Conv2D(filters=filters_list[2], kernel_size=1, strides=1, use_bias=False, name='%s_proj'%(name))(x)
    x = BatchNormalization(name='%s_proj_bn'%(name))(x)

    out = Add(name='%s_add'%(name))([x, out])
    out = ReLU(name='%s_relu'%(name))(out)    
    return out

def resnet_body(x, filters_list, num_blocks, strides=2, name=None):
    out = res_block_proj(x=x, filters_list=filters_list, strides=strides, name='%s_1'%(name))
    for i in range(1, num_blocks):
        out = res_block(x=out, filters_list=filters_list, name='%s_%s'%(name, str(i+1)))
    return out

def detnet_body(x, filters_list, num_blocks, strides=1, name=None):
    out = dilated_res_block_proj(x=x, filters_list=filters_list, name='%s_1'%(name))
    for i in range(1, num_blocks):
        out = dilated_res_block(x=out, filters_list=filters_list, name='%s_%s'%(name, str(i+1)))
    return out


def detnet_59(inputs, filters_list, blocks_list, num_classes):
    # stage 1
    inputs_pad = ZeroPadding2D(padding=3, name='inputs_pad')(inputs)
    conv1 = Conv2D(filters=filters_list[0][0], kernel_size=7, strides=2, use_bias=False, name='conv1')(inputs_pad)
    conv1 = BatchNormalization(name='conv1_bn')(conv1)
    conv1 = ReLU(name='conv1_relu')(conv1)

    # stage 2
    conv1_pad = ZeroPadding2D(padding=1, name='conv1_pad')(conv1)
    conv1_pool = MaxPooling2D(pool_size=3, strides=2, name='conv1_maxpool')(conv1_pad)
    conv2_x = resnet_body(x=conv1_pool, filters_list=filters_list[1], num_blocks=blocks_list[1], strides=1, name='res2')

    # stage 3
    conv3_x = resnet_body(x=conv2_x, filters_list=filters_list[2], num_blocks=blocks_list[2], strides=2, name='res3')

    # stage 4
    conv4_x = resnet_body(x=conv3_x, filters_list=filters_list[3], num_blocks=blocks_list[3], strides=2, name='res4')

    # stage 5
    conv5_x = detnet_body(x=conv4_x, filters_list=filters_list[4], num_blocks=blocks_list[4], strides=1, name='dires5')

    # stage 6
    conv6_x = detnet_body(x=conv5_x, filters_list=filters_list[5], num_blocks=blocks_list[5], strides=1, name='dires6')

    out = AveragePooling2D(pool_size=14, strides=1, name='final_avepool')(conv6_x)
    out = Flatten(name='flatten')(out)
    outputs = Dense(units=num_classes, activation="softmax", kernel_initializer='he_normal', name='dense')(out)
    model = Model(inputs=inputs, outputs=outputs) 
    return model

if __name__=='__main__':
    # create a keras model detnet_59
    inputs = Input(shape=(224,224,3), name='inputs')
    filters_list=[[64],
                  [64,64,256],
                  
                  [128,128,512],
                  [256,256,1024],
                  [256,256,256],
                  [256,256,256]]
    blocks_list=[1,3,4,6,3,3]
    num_classes = 1000
    model = detnet_59(inputs, filters_list, blocks_list, num_classes)

    # check the model
    model.summary()
    plot_model(model, to_file='detnet59.png')
    model.compile(optimizer='sgd', loss='categorical_crossentropy')
    print('Model Compiled')


    











