import tensorflow as tf
import numpy as np

# Load SavedModel
model_path = "mnist_keras"
model = tf.keras.models.load_model(model_path)

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = tf.cast((x_test / 255.0)[..., np.newaxis], tf.float32)  # Normalize -> add channel dimension -> cast to float32


### Test type 1: Get predictions w/o compiling the model
predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)
accuracy = np.mean(predicted_classes == y_test)
print(f"[TEST 1]: Test Accuracy: {accuracy * 100:.2f}%")


### Test type 2: Test the model w/ evaluation so compilation is required
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"[TEST 2]: Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc * 100:.2f}%")

