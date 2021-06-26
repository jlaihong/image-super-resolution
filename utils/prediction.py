import tensorflow as tf
import numpy as np
from PIL import Image


def unpad_patched_scaled_image(lr_image, patches_per_row, patches_per_col, patch_size, padded_scaled_image):
    _, lr_rows, lr_cols, _ = lr_image.shape

    padded_rows = patches_per_row * patch_size
    padded_cols = patches_per_col * patch_size

    row_padding = padded_rows - lr_rows
    col_padding = padded_cols - lr_cols

    scaling_factor = padded_scaled_image.shape[0] // padded_rows

    extract_from_row = int(row_padding / 2 * scaling_factor)
    extract_till_row = -1

    if row_padding > 0:
        extract_till_row = - extract_from_row

    extract_from_col = int(col_padding / 2 * scaling_factor)
    extract_till_col = -1

    if col_padding > 0:
        extract_till_col = - extract_from_col

    return padded_scaled_image[extract_from_row:extract_till_row, extract_from_col:extract_till_col]


def get_sr_image(model, lr, max_image_dimension=512, patch_size=96, batch_size=16):
    lr = lr[np.newaxis]

    if max(lr.shape) <= max_image_dimension:
        return process_full_image(model, lr)

    return process_in_patches(model, lr, patch_size, batch_size)


def process_in_patches(model, lr, patch_size, batch_size):
    extracted_patches = tf.image.extract_patches(images=lr,
                             sizes=[1, patch_size, patch_size, 1],
                             strides=[1, patch_size, patch_size, 1],
                             rates=[1, 1, 1, 1],
                             padding='SAME')

    patches_per_row, patches_per_col = extracted_patches.shape[1], extracted_patches.shape[2]

    extracted_patches = extracted_patches.numpy().reshape(-1, patch_size, patch_size, 3)

    patch_results = []

    for start_index in range(0, extracted_patches.shape[0], batch_size):
        end_index = start_index + batch_size
        
        preds = model.predict(extracted_patches[start_index: end_index])
        
        patch_results.append(preds)

    patch_results = np.concatenate(patch_results, axis=0)
    patch_results = np.clip(patch_results, 0, 255)
    patch_results = np.round(patch_results)

    joined = join_patches(patch_results, patches_per_row, patches_per_col).astype(np.uint8)

    return Image.fromarray(unpad_patched_scaled_image(lr, patches_per_row, patches_per_col, patch_size, joined))


def process_full_image(model, lr):
    sr = model.predict(lr)[0]
    sr = tf.clip_by_value(sr, 0, 255)
    sr = tf.round(sr)
    sr = tf.cast(sr, tf.uint8)
    return Image.fromarray(sr.numpy())


def join_patches(patches, patches_per_row, patches_per_col):
    patch_rows, patch_cols, channels = patches[0].shape
    
    image_rows = patch_rows * (patches_per_row - 1) + patch_rows
    image_cols = patch_cols * (patches_per_col - 1) + patch_cols
    
    joined = np.zeros((image_rows, image_cols, 3))
    
    row, col = 0, 0
    
    for patch in patches:
        joined[row * patch_rows: (row+1) * (patch_rows), col * patch_cols: (col+1) * (patch_cols)] = patch
        
        col += 1
        
        if col >= patches_per_col:
            col = 0
            row += 1
            
    return joined