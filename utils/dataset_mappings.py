import tensorflow as tf


def random_crop(lr_img, hr_img, hr_crop_size, scale):
    lr_crop_size = hr_crop_size // scale
    lr_shape = tf.shape(lr_img)[:2]

    lr_top = tf.random.uniform(shape=(), maxval=lr_shape[0] - lr_crop_size + 1, dtype=tf.int32)
    lr_left = tf.random.uniform(shape=(), maxval=lr_shape[1] - lr_crop_size + 1, dtype=tf.int32)

    hr_top = lr_top * scale
    hr_left = lr_left * scale

    lr_crop = lr_img[lr_top:lr_top + lr_crop_size, lr_left:lr_left + lr_crop_size]
    hr_crop = hr_img[hr_top:hr_top + hr_crop_size, hr_left:hr_left + hr_crop_size]

    return lr_crop, hr_crop


def random_flip(lr_img, hr_img):
    flip_chance = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(flip_chance < 0.5,
                   lambda: (lr_img, hr_img),
                   lambda: (tf.image.flip_left_right(lr_img),
                            tf.image.flip_left_right(hr_img)))


def random_rotate(lr_img, hr_img):
    rotate_option = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr_img, rotate_option), tf.image.rot90(hr_img, rotate_option)


def random_lr_jpeg_noise(lr_img, hr_img, min_jpeg_quality=50, max_jpeg_quality=95):
    jpeg_noise_chance = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(jpeg_noise_chance < 0.5,
                   lambda: (lr_img, hr_img),
                   lambda: (tf.image.random_jpeg_quality(lr_img, min_jpeg_quality, max_jpeg_quality),
                            hr_img))