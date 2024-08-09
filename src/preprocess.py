from keras.utils import image_dataset_from_directory

from config import train_directory, test_directory, image_size, batch_size, validation_split, train_trans_directory, test_trans_directory

def _split_data(train_directory, test_directory, batch_size, validation_split):
    print('train dataset:')
    train_dataset, validation_dataset = image_dataset_from_directory(
        train_directory,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset="both",
        seed=47
    )
    print('test dataset:')
    test_dataset = image_dataset_from_directory(
        test_directory,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False
    )

    return train_dataset, validation_dataset, test_dataset

def get_datasets():
    train_dataset, validation_dataset, test_dataset = \
        _split_data(train_directory, test_directory, batch_size, validation_split)
    return train_dataset, validation_dataset, test_dataset

def get_transfer_datasets(): #animal_crossing_directory, doom_directory, batch_size, validation_split
    # Your code replaces this by loading the dataset
    # you can use image_dataset_from_directory, similar to how the _split_data function is using it
        
    train_trans_directory = 'kaggle/dogcat/train'
    validation_trans_directory = 'kaggle/dogcat/validation'
    test_trans_directory = 'kaggle/dogcat/test1'

    #name/variable to return 
    print('train dogs dataset:')
    train_trans_dataset = image_dataset_from_directory(
        train_trans_directory, # the path
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset="both",
        seed=47
    )
    print('validate dogs dataset:')
    validation_trans_dataset = image_dataset_from_directory(
        validation_trans_directory, # the path
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset="both",
        seed=47
    )
    print('test dogs dataset:')
    test_trans_dataset = image_dataset_from_directory(
        test_trans_directory,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False
    )

    return train_trans_dataset, validation_trans_dataset, test_trans_dataset