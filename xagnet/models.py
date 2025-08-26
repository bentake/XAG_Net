import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Input, Activation, Add, BatchNormalization, multiply, Softmax
from tensorflow.keras.models import Model

def cross_slice_attention(inputs):
    num_slices = inputs.shape[-1]
    att = Conv2D(num_slices, (1,1), padding="same")(inputs)
    att = Softmax(axis=-1)(att)
    weighted = multiply([inputs, att])
    return Add()([inputs, weighted])

def attention_block(x, g, inter_channel):
    theta_x = Conv2D(inter_channel, 1, padding="same")(x)
    phi_g = Conv2D(inter_channel, 1, padding="same")(g)
    f = Activation("relu")(Add()([theta_x, phi_g]))
    psi = Conv2D(1, 1, padding="same")(f)
    rate = Activation("sigmoid")(psi)
    return multiply([x, rate])

def build_xagnet_unet(input_shape=(256,256,3), num_classes=1):
    inp = Input(input_shape)
    refined = cross_slice_attention(inp)

    inp = Input(input_shape)
    refined = cross_slice_attention(inp)

    # Encoder: 64, 128, 256, 512
    c1 = Conv2D(base_filters, 3, padding="same", activation="relu")(refined); c1 = BatchNormalization()(c1); p1 = MaxPooling2D(2)(c1)
    c2 = Conv2D(base_filters*2, 3, padding="same", activation="relu")(p1);    c2 = BatchNormalization()(c2); p2 = MaxPooling2D(2)(c2)
    c3 = Conv2D(base_filters*4, 3, padding="same", activation="relu")(p2);    c3 = BatchNormalization()(c3); p3 = MaxPooling2D(2)(c3)
    c4 = Conv2D(base_filters*8, 3, padding="same", activation="relu")(p3);    c4 = BatchNormalization()(c4); p4 = MaxPooling2D(2)(c4)

    # Bottleneck: 1024
    bn = Conv2D(base_filters*16, 3, padding="same", activation="relu")(p4);   bn = BatchNormalization()(bn)

    # Decoder: 512, 256, 128, 64 (with AG + CSA at each skip)
    u5 = Conv2DTranspose(base_filters*8, 3, strides=2, padding="same", activation="relu")(bn)
    a5 = attention_block(c4, u5, base_filters*8); cs5 = cross_slice_attention(c4)
    u5 = concatenate([u5, a5, cs5]); u5 = Conv2D(base_filters*8, 3, padding="same", activation="relu")(u5); u5 = BatchNormalization()(u5)

    u6 = Conv2DTranspose(base_filters*4, 3, strides=2, padding="same", activation="relu")(u5)
    a6 = attention_block(c3, u6, base_filters*4); cs6 = cross_slice_attention(c3)
    u6 = concatenate([u6, a6, cs6]); u6 = Conv2D(base_filters*4, 3, padding="same", activation="relu")(u6); u6 = BatchNormalization()(u6)

    u7 = Conv2DTranspose(base_filters*2, 3, strides=2, padding="same", activation="relu")(u6)
    a7 = attention_block(c2, u7, base_filters*2); cs7 = cross_slice_attention(c2)
    u7 = concatenate([u7, a7, cs7]); u7 = Conv2D(base_filters*2, 3, padding="same", activation="relu")(u7); u7 = BatchNormalization()(u7)

    u8 = Conv2DTranspose(base_filters, 3, strides=2, padding="same", activation="relu")(u7)
    a8 = attention_block(c1, u8, base_filters);   cs8 = cross_slice_attention(c1)
    u8 = concatenate([u8, a8, cs8]); u8 = Conv2D(base_filters, 3, padding="same", activation="relu")(u8); u8 = BatchNormalization()(u8)

    out = Conv2D(num_classes, 1, activation="sigmoid")(u8)
    return Model(inp, out)

def compile_model(model, learning_rate=1e-3, loss="binary_crossentropy", metrics=None):
    if metrics is None:
        metrics = []
    opt = tf.keras.optimizers.Adam(learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model
