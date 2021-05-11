import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_dataset_iterator(dataset_dir, batch_size=8, target_size=(150, 150)):
    datagen = ImageDataGenerator(1/255.,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    return datagen.flow_from_directory(dataset_dir,
                                                  batch_size=batch_size,
                                                  target_size=target_size,
                                 class_mode='sparse')

def get_simple_classifier(input_shape=(150, 150, 3)):

    return tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                   input_shape=input_shape),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(4, activation='softmax')
    ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple ML pipeline for '\
                                     'classifiers training')
    parser.add_argument('dataset_dir', type=str, help="Dataset directory")
    parser.add_argument('--learning-rate', type=float, help="Learning rate",
                    default=0.0001)
    parser.add_argument('--batch-size', type=int, help="Batch size",
                    default=1)
    parser.add_argument('--patch-size', type=int, help="Patch size",
                    default=150)
    parser.add_argument('--epochs', type=int, help="Epochs",
                    default=150)

    args = parser.parse_args()
    args = vars(args)

    tgen = get_dataset_iterator(args['dataset_dir'],
                                batch_size=args['batch_size'],
                                target_size=(args['patch_size'],
                                             args['patch_size']))

    for x, i in tgen:
        print(x.shape)
        print(i.shape)
        break

    model = get_simple_classifier()
    model.summary()

    model.compile(optimizer=RMSprop(lr=args['learning_rate']), loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    history = model.fit(tgen, epochs=args['epochs'])
    print(history)
