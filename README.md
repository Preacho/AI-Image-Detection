# AI-Image-Detection Analysis

This project aims to determine effective models that can discern AI-Generated Images from a random set of Images. By training the models with a diverse set of image datasets, high accurate predictions were achieved using different machine learning models. 

## Built With

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

## 1. Installation

To install and reproduce this project, follow these steps:

1. Clone the repository
   
``` bash
git clone https://github.com/Preacho/AI-Image-Detection-Model.git 

```


2. Install python packages using the following command

``` bash
pip install -r requirements.txt
```
3. Install dataset from the following link [https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images/data]. Move the dataset into the repository.


## 2. Demo 

Run to collect a processed dataset 
``` bash
python src/image_preprocess.py
```


After:
``` bash
python src\evaluate_model.py ## to train the zipped models
python src\training_image_detection_model__nn.py ## neural networks
python src\training_image_detection_model_lightgbm.py ##lightgbm 
python src\training_image_detection_model_randomforest.py ##random forest

```

## Report
[View the full analysis report](https://preacho.github.io/AI-Image-Detection-Analysis/)
