from tensorflow.keras.layers import Conv1D, SpatialDropout1D, Activation, add, concatenate

def residual_block(x, dilation_rate, nb_filters, kernel_size, padding, dropout_rate=0):
    prev_x = x

    ## WHY ARE WE LOOPING K OVER RANGE(2)????
    for k in range(2): 
        x = Conv1D(filters=nb_filters,
                   kernel_size=kernel_size,
                   dilation_rate=dilation_rate,
                   padding=padding)(x)
        x = Activation('relu')(x)
        x = SpatialDropout1D(rate=dropout_rate)(x)

    # 1x1 conv to match the shapes (channel dimension).
    prev_x = Conv1D(nb_filters, 1, padding='same')(prev_x)
    res_x = add([prev_x, x])
    return res_x, x


