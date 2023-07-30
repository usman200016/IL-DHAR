import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical


def build_decision_tree(inputs, num_classes):
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    tree_model = models.Model(inputs, x)

    return tree_model
def build_dndf(input_shape, num_classes, num_trees=10):
    inputs = layers.Input(shape=input_shape)
    trees = []
    for _ in range(num_trees):
        tree = build_decision_tree(inputs, num_classes)
        trees.append(tree)
    forest_output = layers.Average()(trees)

    outputs = layers.Dense(num_classes, activation='softmax')(forest_output)
    model = models.Model(inputs, outputs)

    return model

def train_dndf(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=10):
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))

    return model