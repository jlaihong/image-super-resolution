import os
import tensorflow as tf
import glob
from tensorflow.python.data.experimental import AUTOTUNE

hr_train_url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
hr_valid_url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"


def download_data(download_url, data_directory):
    file = download_url.split("/")[-1]
    print(data_directory)
    tf.keras.utils.get_file(file, download_url, cache_subdir=data_directory, extract=True)
    os.remove(os.path.join(data_directory, file))
    

def image_dataset_from_directory_or_url(data_directory, image_directory, download_url):
    images_path = os.path.join(data_directory, image_directory)

    if not os.path.exists(images_path):
        print("Couldn't find directory: ", images_path)
        os.makedirs(data_directory, exist_ok=True)
        download_data(download_url, data_directory)

    filenames = sorted(glob.glob(images_path + "/*.png"))

    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(tf.io.read_file)
    dataset = dataset.map(lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE)

    cache_directory = os.path.join(data_directory, "cache", image_directory)

    os.makedirs(cache_directory, exist_ok=True)

    cache_file = cache_directory + "/cache"

    dataset = dataset.cache(cache_file)

    if not os.path.exists(cache_file + ".index"):
        populate_cache(dataset, cache_file)

    return dataset


def create_training_dataset(dataset_parameters, train_mappings, batch_size):
    lr_dataset = image_dataset_from_directory_or_url(dataset_parameters.save_data_directory, dataset_parameters.train_directory, dataset_parameters.train_url)
    hr_dataset = image_dataset_from_directory_or_url(dataset_parameters.save_data_directory, "DIV2K_train_HR", hr_train_url)

    dataset = tf.data.Dataset.zip((lr_dataset, hr_dataset))

    for mapping in train_mappings:
        dataset = dataset.map(mapping, num_parallel_calls=AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset


def create_validation_dataset(dataset_parameters):
    lr_dataset = image_dataset_from_directory_or_url(dataset_parameters.save_data_directory, dataset_parameters.valid_directory, dataset_parameters.valid_url)
    hr_dataset = image_dataset_from_directory_or_url(dataset_parameters.save_data_directory, "DIV2K_valid_HR", hr_valid_url)

    dataset = tf.data.Dataset.zip((lr_dataset, hr_dataset))

    dataset = dataset.batch(1)
    dataset = dataset.repeat(1)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset


def create_training_and_validation_datasets(dataset_parameters, train_mappings, train_batch_size=16):
    training_dataset = create_training_dataset(dataset_parameters, train_mappings, train_batch_size)
    validation_dataset = create_validation_dataset(dataset_parameters)

    return training_dataset, validation_dataset


def populate_cache(dataset, cache_file):
    print(f'Begin caching in {cache_file}.')
    for _ in dataset: pass
    print(f'Completed caching in {cache_file}.')

