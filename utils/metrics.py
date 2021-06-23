import tensorflow as tf

def psnr_metric(x1, x2):
    return tf.image.psnr(x1, x2, max_val=255)