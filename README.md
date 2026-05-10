# cs4824 ML Model Final Project
Final Project for CS4824 ML Project by 

Nathan Le and Zachary Babka

## Project Description
Design and evaluate machine learning models to automatically classify biomedical images for Pneumonia. The project focuses on translating raw image data into clinically meaningful predictions.

## Requirements

Python 3+

[MedMNIST](https://pypi.org/project/medmnist/)
```python
pip install medmnist
```
This will install all additional required packages such as PyTorch, Numpy, etc

It is recommend to install medmnist in a virtual environment to avoid conflicting installs


# Running the program
Once cloned down and with all required packages installed, simply run the main.

```python
python main.py
```
This will run the program, downloading all the necessary training data.

The model can be configured to specification such as:
- Changing Seed
- Changing Image size (28x28, 64x64, 128x128)
- Epoch Amount

