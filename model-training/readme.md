# Install

## TensorFlow & Keras (with specific version)
https://keras.io/getting_started/
TensorFlow compatibility
To use Keras 2:
tensorflow~=2.15.0 & keras~=2.15.0

```sh
pip install tensorflow~=2.15.0 keras~=2.15.0
```

## Nobuco
```sh
pip install -U nobuco
```


## Others
```sh
pip install torch torchvision matplotlib
```


# Scripts
- `model.py`: Define the model in `PyTorch`
- `train.py`: Train the model and save to `.pth` file
- `format.py`: Load the `.pth` file and save to `SavedModel` format
- `tf_test.py`: Load the `SavedModel` and test with TensorFlow

