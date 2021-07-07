from tensorflow.keras import Input
from tensorflow.keras.layers import Lambda, add
from tensorflow.keras.models import Model
import idac_tcn as tcn


def network(pdict, padding='causal', drop=0.05):
    
    nb_chan    = len([char for char in pdict['cmpts']])
    nb_filters = pdict['f']
    filter_len = pdict['k']
    dilations  = pdict['d']
    nb_stacks  = pdict['s']
    
    
    input_layer = Input(shape=(None, nb_chan))

    x = input_layer

    skip_connections = []
    for s in range(nb_stacks):
        for d in dilations:
            x, skip_out = tcn.residual_block(x,
                                         dilation_rate=d,
                                         nb_filters=nb_filters,
                                         kernel_size=filter_len,
                                         padding=padding,
                                         dropout_rate=drop)
            skip_connections.append(skip_out)

    x = add(skip_connections)
    x = Lambda(lambda tt: tt[:, -1, :])(x)
    
    x = Dense(512, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    
    output_layer = Dense(1, activation='sigmoid', name='output')(x)

    return Model(input_layer, output_layer, name='model')

