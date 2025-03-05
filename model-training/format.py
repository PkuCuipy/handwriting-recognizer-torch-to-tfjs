import torch
import nobuco
from nobuco import ChannelOrder
from model import MNISTModel


model_path = "best_mnist_model.pth"
model = MNISTModel()
model.load_state_dict(torch.load(model_path))
model.eval()

# First convert to Keras format
dummy_input = torch.randn(1, 1, 28, 28)
keras_model = nobuco.pytorch_to_keras(
    model,
    args=[dummy_input], kwargs=None,
    inputs_channel_order=ChannelOrder.TENSORFLOW,
    outputs_channel_order=ChannelOrder.TENSORFLOW
)
keras_model.summary()

# Then Save keras model to SavedModel format
keras_model.save("mnist_keras")
