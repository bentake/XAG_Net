import tensorflow as tf

def dice_coef(y_true, y_pred, smooth: float = 1.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    denom = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    return (2.0 * intersection + smooth) / (denom + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def boundary_loss(y_true, y_pred):
    # ensure (B,H,W,1)
    if y_true.shape.rank == 3:
        y_true = tf.expand_dims(y_true, -1)
    if y_pred.shape.rank == 3:
        y_pred = tf.expand_dims(y_pred, -1)
    gt = tf.image.sobel_edges(tf.cast(y_true, tf.float32))
    pr = tf.image.sobel_edges(tf.cast(y_pred, tf.float32))
    return tf.reduce_mean(tf.abs(gt - pr))

def combined_validation_loss(y_true, y_pred):
    return 0.9 * dice_loss(y_true, y_pred) + 0.1 * boundary_loss(y_true, y_pred)
