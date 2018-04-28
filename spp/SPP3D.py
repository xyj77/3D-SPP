from keras.engine.topology import Layer
import keras.backend as K
from keras.layers.convolutional import MaxPooling3D
from keras.layers.core import Flatten


class SPP3D(Layer):
    """Spatial pyramid pooling layer for 3D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_list: list of int
            List of pooling regions to use. The length of the list is the number of pooling regions,
            each int in the list is the number of regions in that pool. For example [1,2,4] would be 3
            regions with 1, 2x2x2 and 3x3x3 max pools, so 21 outputs per feature map
    # Input shape
        5D tensor with shape:
        `(samples, channels, rows, cols, hight)` if dim_ordering='th'
        or 5D tensor with shape:
        `(samples, rows, cols, hight, channels)` if dim_ordering='tf'.
    # Output shape
        2D tensor with shape:
        `(samples, channels * sum([i * i for i in pool_list])`
    """

    def __init__(self, pool_list, **kwargs):

        self.dim_ordering = K.image_dim_ordering()
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.pool_list = pool_list

        self.num_outputs_per_channel = sum([i*i*i for i in pool_list])

        super(SPP3D, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[1]
        elif self.dim_ordering == 'tf':
            self.nb_channels = input_shape[4]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.nb_channels * self.num_outputs_per_channel)

    def get_config(self):
        config = {'pool_list': self.pool_list}
        base_config = super(SPP3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):

        input_shape = K.shape(x)

        if self.dim_ordering == 'th':
            num_rows = input_shape[2]
            num_cols = input_shape[3]
            num_hight = input_shape[4]
        elif self.dim_ordering == 'tf':
            num_rows = input_shape[1]   #8
            num_cols = input_shape[2]   #16
            num_hight = input_shape[3]  #24

        row_length = [K.cast(num_rows, 'float32') / i for i in self.pool_list]     #[8,4,2]
        col_length = [K.cast(num_cols, 'float32') / i for i in self.pool_list]     #[16,8,4]
        hight_length = [K.cast(num_hight, 'float32') / i for i in self.pool_list]  #[24,12,6]

        outputs = []

        if self.dim_ordering == 'th':
            for pool_num, num_pool_regions in enumerate(self.pool_list):
                for ix in range(num_pool_regions):
                    for jy in range(num_pool_regions):
                        for kz in range(num_pool_regions):
                            x1 = ix * row_length[pool_num]
                            x2 = ix * row_length[pool_num] + row_length[pool_num]
                            y1 = jy * col_length[pool_num]
                            y2 = jy * col_length[pool_num] + col_length[pool_num]
                            z1 = kz * hight_length[pool_num]
                            z2 = kz * hight_length[pool_num] + hight_length[pool_num]

                            x1 = K.cast(K.round(x1), 'int32')
                            x2 = K.cast(K.round(x2), 'int32')
                            y1 = K.cast(K.round(y1), 'int32')
                            y2 = K.cast(K.round(y2), 'int32')
                            z1 = K.cast(K.round(z1), 'int32')
                            z2 = K.cast(K.round(z2), 'int32')
                            new_shape = [input_shape[0], input_shape[1],
                                         x2 - x1, y2 - y1, z2 - z1]
                            x_crop = x[:, :, z1:z2, y1:y2, x1:x2]
                            xm = K.reshape(x_crop, new_shape)
                            pooled_val = K.max(xm, axis=(2, 3, 4))
                            outputs.append(pooled_val)

        elif self.dim_ordering == 'tf':
            for pool_num, num_pool_regions in enumerate(self.pool_list):   #[0:1 1:2 2:4]
                #####################
                # outputs = MaxPooling3D(pool_size=(row_length[pool_num],col_length[pool_num],hight_length[pool_num]))(x)
                # outputs = Flatten()(outputs)
                #####################
                for ix in range(num_pool_regions): #[1,2,4]
                    for jy in range(num_pool_regions): #[1,2,4]
                        for kz in range(num_pool_regions): #[1,2,4]
                            x1 = ix * row_length[pool_num] #1x8  2x4  4x2
                            x2 = ix * row_length[pool_num] + row_length[pool_num]
                            y1 = jy * col_length[pool_num]
                            y2 = jy * col_length[pool_num] + col_length[pool_num]
                            z1 = kz * hight_length[pool_num]
                            z2 = kz * hight_length[pool_num] + hight_length[pool_num]

                            x1 = K.cast(K.round(x1), 'int32')
                            x2 = K.cast(K.round(x2), 'int32')
                            y1 = K.cast(K.round(y1), 'int32')
                            y2 = K.cast(K.round(y2), 'int32')
                            z1 = K.cast(K.round(z1), 'int32')
                            z2 = K.cast(K.round(z2), 'int32')

                            new_shape = [input_shape[0], x2 - x1, y2 - y1,
                                         z2 - z1, input_shape[4]]

                            x_crop = x[:, x1:x2, y1:y2, z1:z2, :]
                            xm = K.reshape(x_crop, new_shape)
                            pooled_val = K.max(xm, axis=(1, 2, 3))
                            outputs.append(pooled_val)

        if self.dim_ordering == 'th':
            outputs = K.concatenate(outputs)
        elif self.dim_ordering == 'tf':
            #outputs = K.concatenate(outputs,axis = 1)
            outputs = K.concatenate(outputs)
            #outputs = K.reshape(outputs,(len(self.pool_list),self.num_outputs_per_channel,input_shape[0],input_shape[1]))
            #outputs = K.permute_dimensions(outputs,(3,1,0,2))
            #outputs = K.reshape(outputs,(input_shape[0], self.num_outputs_per_channel * self.nb_channels))

        return outputs
