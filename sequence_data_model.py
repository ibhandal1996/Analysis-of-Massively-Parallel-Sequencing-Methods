import pandas as pd
import tensorflow as tf


def input_fn(features, labels, training=True, batch_size=256):
    # Converts the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffles and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)


def main():
    # Loads the dataset for training and testing
    x = pd.read_csv("training_data.csv")
    y = x.iloc[:, -1]
    x = x.iloc[:, :-1]

    # Splits up data for training and testing.
    x_train = x.iloc[:750]
    x_test = x.iloc[750:]
    old_y_train = y.iloc[:750]
    old_y_test = y.iloc[750:]

    columns_numerical = ['A_1', 'C_1', 'G_1', 'T_1']
    key = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}

    # Converts A, C, G, T, N values into numerical values using key dictionary.
    temp = []
    for letter in old_y_train:
        temp.append(key[letter])

    y_train = pd.DataFrame(temp)
    temp = []
    for letter in old_y_test:
        temp.append(key[letter])

    y_test = pd.DataFrame(temp)

    # Creates a feature columns list for the model.
    my_feature_columns = []
    for key in columns_numerical:
        my_feature_columns.append(tf.feature_column.numeric_column(key, dtype=tf.float32))
    print(my_feature_columns)

    # Builds a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
    classifier = tf.estimator.LinearClassifier(
        feature_columns=my_feature_columns,
        # The model must choose between 5 classes.
        n_classes=5)

    # Trains the model.
    classifier.train(
        input_fn=lambda: input_fn(x_train, y_train, training=True),
        steps=5000)

    # Tests the model.
    test_result = classifier.evaluate(
        input_fn=lambda: input_fn(x_test, y_test, training=False))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**test_result))

    # Creates feature columns for the exported model.
    A = tf.feature_column.numeric_column('A_1')
    C = tf.feature_column.numeric_column('C_1')
    G = tf.feature_column.numeric_column('G_1')
    T = tf.feature_column.numeric_column('T_1')

    # Creates an input function for the exported model.
    serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        tf.feature_column.make_parse_example_spec(my_feature_columns))

    # Exports the model.
    classifier.export_saved_model("D:\CompleteGenomics\myModel", serving_input_receiver_fn=serving_input_fn)


if __name__ == "__main__":
    main()
