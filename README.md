# Iris Dectection Model Training with TensorFlow

(`train_model.py`) trains an iris detection model using TensorFlow. The model is trained on a user's dataset and predicts detects the location of both irises.

## Prerequisites

Before running (`train_model.py`),  make sure you have are using **Python 3.9** and have the required python libraries installed:

- TensorFlow (2.10)
- Matplotlib

**NOTE**:
To train this model, you will need to create your own dataset.
For crafting your dataset, I reccommend the following modules:

  `OpenCv` for collecting images<br>
  `labelme` for annotating the irises in each photo<br>
  `albumentations` for augmenting the original pictures in order to create more data<br>

After the dataset is completed, the filepaths utilzed in (`train_model.py`) will need to be changed according to YOUR file path.

You can install the dependencies using the following command:
```bash
pip install tensorflow cv2 albumentations labelme matplotlib
