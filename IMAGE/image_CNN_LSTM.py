import os
import json

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import dropout
from tqdm import tqdm
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Flatten, TimeDistributed, MaxPooling1D, Dropout,  Bidirectional, Conv2D,MaxPool2D


def main():
    data_root = os.path.abspath(os.path.join(os.getcwd(), "./"))  # get data root path
    image_path = os.path.join(data_root, "data")
    train_dir = os.path.join(image_path, "byte_png_train")
    validation_dir = os.path.join(image_path, "byte_png_val")
    assert os.path.exists(train_dir), "cannot find {}".format(train_dir)
    assert os.path.exists(validation_dir), "cannot find {}".format(validation_dir)

    im_height = 256
    im_width = 256
    batch_size = 16
    epochs = 35
    num_classes = 9

    train_image_generator = ImageDataGenerator(rescale=1/255)

    validation_image_generator = ImageDataGenerator(rescale=1/255)

    train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                               batch_size=batch_size,
                                                               shuffle=True,
                                                               color_mode='grayscale',
                                                               target_size=(im_height, im_width),
                                                               class_mode='categorical')
    total_train = train_data_gen.n

    # get class dict
    class_indices = train_data_gen.class_indices

    # transform value and key of dict
    inverse_dict = dict((val, key) for key, val in class_indices.items())
    # write dict into json file
    json_str = json.dumps(inverse_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                                  batch_size=batch_size,
                                                             shuffle=False,
                                                                  target_size=(im_height, im_width),
                                                                  color_mode='grayscale',
                                                                  class_mode='categorical')
    # img, _ = next(train_data_gen)
    total_val = val_data_gen.n
    print("using {} images for training, {} images for validation.".format(total_train,
                                                                           total_val))

    model = tf.keras.Sequential([
        TimeDistributed(Conv1D(64, 5, input_shape=(im_height, im_width, 1), activation='relu')),
        TimeDistributed(MaxPooling1D(pool_size=4)),
        TimeDistributed(Conv1D(128, 4, activation='relu')),
        TimeDistributed(MaxPooling1D(pool_size=3)),
        TimeDistributed(Dropout(0.5)),
        TimeDistributed(Flatten()),
        TimeDistributed(Dense(256)),
        Bidirectional(LSTM(256, name='lstm_layer', return_sequences=True) ),
        Bidirectional(LSTM(128)),
        # LSTM(70, recurrent_dropout=0.5),
        Dense(128, activation='relu'),
        Dense(num_classes,activation='softmax')
    ])

    # model = tf.keras.Sequential([
    #     Conv2D(64,3,input_shape=(im_height,im_width,1), activation='relu'),
    #     MaxPool2D(pool_size=(3,3)),
    #     Conv2D(128,3,activation='relu'),
    #     MaxPool2D(pool_size=(2,2)),
    #     Dropout(0.5),
    #     Flatten(),
    #     Dense(128),
    #     Dense(num_classes, activation='softmax') 
    # ])

    # using keras low level api for training
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')

    res_file = open('./image_CNN_LSTM_trainsteps.txt', mode='a')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            output = model(images, training=True)
            loss = loss_object(labels, output)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, output)

    @tf.function
    def val_step(images, labels):
        output = model(images, training=False)
        loss = loss_object(labels, output)

        val_loss(loss)
        val_accuracy(labels, output)

    best_val_acc = 0.
    for epoch in range(epochs):
        train_loss.reset_states()  # clear history info
        train_accuracy.reset_states()  # clear history info
        val_loss.reset_states()  # clear history info
        val_accuracy.reset_states()  # clear history info

        # train
        train_bar = tqdm(range(total_train // batch_size))
        for step in train_bar:
            images, labels = next(train_data_gen)
            train_step(images, labels)

            # print train process
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                                 epochs,
                                                                                 train_loss.result(),
                                                                                 train_accuracy.result())
            
        res_file.write("train epoch[{}/{}] loss:{:.3f}, acc:{:.3f}\n".format(epoch + 1,
                                                                                 epochs,
                                                                                 train_loss.result(),
                                                                                 train_accuracy.result()))

        # validate
        val_bar = tqdm(range(total_val // batch_size))
        for step in val_bar:
            test_images, test_labels = next(val_data_gen)
            val_step(test_images, test_labels)

            # print val process
            val_bar.desc = "valid epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                               epochs,
                                                                               val_loss.result(),
                                                                               val_accuracy.result())
        
        res_file.write("valid epoch[{}/{}] loss:{:.3f}, acc:{:.3f}\n".format(epoch + 1,
                                                                                 epochs,
                                                                                 val_loss.result(),
                                                                                 val_accuracy.result()))
        res_file.flush()
        # only save best weights
        if val_accuracy.result() > best_val_acc:
            best_val_acc = val_accuracy.result()
            model.save_weights("./save_weights/byte_cnn_lstm.ckpt", save_format="tf")
    res_file.close()  


if __name__ == '__main__':
    main()
