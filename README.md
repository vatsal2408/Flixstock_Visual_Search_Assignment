# Flixstock_Visual_Search_Assignment
## About the Project
We often see online shopping sites suggesting clothes based on our preference of design and type.
This project aims to find garments of similar pattern or type given a query image by matching features of query image with the dataset.
To extract features, it primarily uses Google's _Inception-V3_ in evaluation mode but other options are available for experimentation.

## Prerequisites
To install the required packages/modules, run below code:
```
pip install -r requirements.txt
```

## Usage
To run this on your own system, download data from [here]() and paste it inside _data_ folder then select the query image, **path/query.jpg**, and run following code-block:
```
python main.py --input_image path/query.jpg
```
To speed-up creation of embeddings, you can use your gpu, if available, by using `device` (default is _cpu_):
```
python main.py --input_image path/query.jpg --device cuda
```

This project creates embedding only the first time, so subsequent runs would be considerably faster. It automatically checks for changes in dataset to create new embeddings if required or if other model is used instead of Inception network.

If you want to use your own images, specify your folder path **path/folder** by using `img_folder`:
```
python main.py --input_image path/query.jpg --img_folder path/folder
```
If you want to experiment with models other than Inception V3, you can use `model` to pass your desired model (options available are vgg16, vgg19, resnet18, resnet34):
```
python main.py --input_image path/query.jpg --model resnet18
```

## Results
Here are few outputs of similar images when passed a query image:
