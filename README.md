# Flixstock Visual Search Assignment
## About the Project

<p align="center">
    <img src="/data/Screenshot (48).png">
</p>

We often see online shopping sites suggesting clothes based on our preference of design and type.
This project aims to find garments of similar pattern or type given a query image by matching features of query image with the dataset.
To extract features, it primarily uses Google's _Inception-V3_ in evaluation mode but other options are available for experimentation.

## Prerequisites
To install the required packages/modules, run below code-block:
```
pip install -r requirements.txt
```

## Usage
To run this on your own system, download dataset from [here](https://drive.google.com/drive/folders/1OjoTr792sA6_wh1OarYulhdaln3RNBC9?usp=sharing) and extract the **folder** in _/data_, then select the query image, _**path/query.jpg**_, and run following code-block:
```
python main.py --input_image path/query.jpg
```
To speed-up creation of embeddings, you can use your gpu, if available, by passing _cuda_ in `device` (default is _cpu_):
```
python main.py --input_image path/query.jpg --device cuda
```

>In, this project, we create embedding only the first time, so subsequent runs would be considerably faster. It automatically checks for changes in dataset to create new embeddings if required or if other model is used instead of Inception network.

If you want to use your own images as dataset, specify your folder path: _**path/folder**_ by passing into `img_folder`:
```
python main.py --input_image path/query.jpg --img_folder path/folder
```
If you want to experiment with models other than Inception V3, you can use `model` to pass your desired model (options available are vgg16, vgg19, resnet18, resnet34):
```
python main.py --input_image path/query.jpg --model resnet18
```

## Results
### Here are few example output when passed a query image:

&emsp; **Query image** &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; **Similar images**

![1](/data/results/13589722RJD.jpg) ![2](/data/results/35468716LXD.jpg) ![3](/data/results/nearestneigh.jpg) ![4](/data/results/13586231PAR.jpg)
![5](/data/results/35464472VTD.jpg)

## Additional approaches

Although this project generated satisfactory results using pre-trained networks, but as images get more complex, we might need to move-on to a network trained on our dataset or we can also train an _Encoder/U-net_ and use its bottleneck layer as feature generator.

I also tried this with `NearestNeighbour` module of `sklearn` and got pretty good rsults. This is available in _nearest.ipynb_ for checking but the directory paths need to be changed accordingly as this was my first rough attempt at this problem.

I believe apart from euclidian and cosine distance, Earth mover's distance might also give good results but I couldn't try that approach in the given timeframe and will update this repo if it gives promising results as well.