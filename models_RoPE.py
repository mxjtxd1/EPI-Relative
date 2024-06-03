# from tensorflow.keras.layers import *
# from tensorflow.keras.models import *
import numpy as np
import tensorflow as tf
from transformer_RoPE import Transformer
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import initializers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization


class AttLayer(Layer):
    def __init__(self, attention_dim):
        # self.init = initializers.get('normal')
        self.init = initializers.RandomNormal(seed=10)
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim,)))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        #        self.trainable_weights = [self.W, self.b, self.u]
        self.trainable_weights.append([self.W, self.b, self.u])
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) +
                      K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def get_model(max_len_en, max_len_pr, nwords, emb_dim):
    enhancers = Input(shape=(max_len_en,))
    promoters = Input(shape=(max_len_pr,))

    embedding_matrix = np.load('embedding_matrix.npy')

    emb_en = Embedding(nwords, emb_dim,
                       weights=[embedding_matrix], trainable=True)(enhancers)
    # print(emb_en)
    emb_pr = Embedding(nwords, emb_dim,
                       weights=[embedding_matrix], trainable=True)(promoters)
    # print(emb_pr)
    enhancer_conv_layer = Conv1D(filters=72,  # 72
                                 kernel_size=36,  # 36
                                 padding="valid",
                                 activation='relu', )(emb_en)
    enhancer_max_pool_layer = MaxPooling1D(pool_size=20, strides=20)(enhancer_conv_layer)
    enhancer_max_pool_layer = enhancer_max_pool_layer[:,:-1,:]
    enhancer_global_max_pool_layer = GlobalMaxPooling1D()(enhancer_conv_layer)
    enhancer_global_max_pool_layer = enhancer_global_max_pool_layer[:, tf.newaxis, :]
    # tf.expand_dims(enhancer_global_max_pool_layer, axis=1)
    enhancer_global_max_pool_layer = tf.cast(enhancer_global_max_pool_layer, tf.float32 )
    enhancer_cnn_layer = tf.concat([enhancer_max_pool_layer,enhancer_global_max_pool_layer],axis=1)
    #l_gru_enhancer_1 = Bidirectional(GRU(40, return_sequences=True))(enhancer_cnn_layer)
    #l_gru_enhancer_2 = BatchNormalization()(l_gru_enhancer_1)
    #l_gru_enhancer_3 = Dropout(0.5)(l_gru_enhancer_2)

    # enhancer_branch =Sequential()
    # enhancer_branch.add(enhancer_conv_layer)
    # enhancer_branch.add(enhancer_max_pool_layer)
    # enhancer_branch.add(BatchNormalization())
    # enhancer_branch.add(Dropout(0.5))
    # enhancer_out = enhancer_branch(emb_en)

    promoter_conv_layer = Conv1D(filters=72,
                                 kernel_size=36,  #
                                 padding="valid",
                                 activation='relu', )(emb_pr)
    promoter_max_pool_layer = MaxPooling1D(pool_size=20, strides=20)(promoter_conv_layer)
    promoter_max_pool_layer = promoter_max_pool_layer[:,:-1,:]
    promoter_global_max_pool_layer = GlobalMaxPooling1D()(promoter_conv_layer)
    promoter_global_max_pool_layer = promoter_global_max_pool_layer[:, tf.newaxis, :]
    promoter_global_max_pool_layer = tf.cast(promoter_global_max_pool_layer, tf.float32 )
    promoter_cnn_layer = tf.concat([promoter_max_pool_layer,promoter_global_max_pool_layer],axis=1)
    #l_gru_promoter_1 = Bidirectional(GRU(40, return_sequences=True))(promoter_cnn_layer)
    #l_gru_promoter_2 = BatchNormalization()(l_gru_promoter_1)
    #l_gru_promoter_3 = Dropout(0.5)(l_gru_promoter_2)

    # promoter_branch = Sequential()
    # promoter_branch.add(promoter_conv_layer)
    # promoter_branch.add(promoter_max_pool_layer)
    # promoter_branch.add(BatchNormalization())
    # promoter_branch.add(Dropout(0.5))
    # promoter_out = promoter_branch(emb_pr)

    # l_gru_1 = Bidirectional(GRU(50, return_sequences=True))(enhancer_out)
    # l_gru_2 = Bidirectional(GRU(50, return_sequences=True))(promoter_out)
    #l_att_1 = AttLayer(36)(l_gru_1)
    #l_att_2 = AttLayer(36)(l_gru_2)

    transformer1 = Transformer(encoder_stack=4,
                               feed_forward_size=256,
                               n_heads=8,
                               model_dim=72)

    transformer2 = Transformer(encoder_stack=4,
                               feed_forward_size=256,
                               n_heads=8,
                               model_dim=72)

    enhancer_trf = transformer1(enhancer_cnn_layer)
    promoter_trf = transformer2(promoter_cnn_layer)

    # enhancer_trf = transformer1(enhancer_max_pool_layer)
    # promoter_trf = transformer2(promoter_max_pool_layer)
    #
    import ipdb
    ipdb.set_trace()
    enhancer_maxpool = GlobalMaxPooling1D()(enhancer_trf)
    promoter_maxpool = GlobalMaxPooling1D()(promoter_trf)

    # merge
    merge = tf.concat([enhancer_maxpool * promoter_maxpool,
                       tf.abs(enhancer_maxpool - promoter_maxpool),
                       ], -1)

    merge1 = BatchNormalization()(merge)
    merge2 = Dropout(0.5)(merge1)

    merge3 = Dense(50, activation='relu')(merge2)

    preds = Dense(1, activation='sigmoid')(merge3)
    model = Model([enhancers, promoters], preds)
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model
