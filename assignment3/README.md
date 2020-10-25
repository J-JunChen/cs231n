## 1. Introduction
The assignment3 includes 5 Questions:

##### Q1: Image Captioning with Vanilla RNNs (30 points)
The notebook **RNN_Captioning.ipynb** will walk you through the implementation of an image captioning system on MS-COCO using vanilla recurrent networks.

##### Q2: Image Captioning with LSTMs (25 points)
The notebook **LSTM_Captioning.ipynb** will walk you through the implementation of Long-Short Term Memory (LSTM) RNNs, and apply them to image captioning on MS-COCO.

##### Q3: Network Visualization: Saliency maps, Class Visualization, and Fooling Image (15 points)
The notebooks **NetworkVisualization-TensorFlow.ipynb**, and **NetworkVisualization-PyTorch**.ipynb will introduce the pretrained SqueezeNet model, compute gradients with respect to images, and use them to produce saliency maps and fooling images. Please complete only one of the notebooks (TensorFlow or PyTorch). No extra credit will be awardeded if you complete both notebooks.

##### Q4: Style Transfer (15 points)
In thenotebooks **StyleTransfer-TensorFlow.ipynb** or **StyleTransfer-PyTorch.ipynb** you will learn how to create images with the content of one image but the style of another. Please complete only one of the notebooks (TensorFlow or PyTorch). No extra credit will be awardeded if you complete both notebooks.

##### Q5: Generative Adversarial Networks (15 points)
In the notebooks **GANS-TensorFlow.ipynb** or **GANS-PyTorch.ipynb** you will learn how to generate images that match a training dataset, and use these models to improve classifier performance when training on a large amount of unlabeled data and a small amount of labeled data. Please complete only one of the notebooks (TensorFlow or PyTorch). No extra credit will be awarded if you complete both notebooks.

## 2. Experiments
### 2.1 Datasets
- #### 2.1.1 Microsoft COCO for Q1 and Q2
- #### 2.1.2 ImageNet for Q3
- #### 2.1.3 MNIST for  Q5

### 2.2 Evaluation Metrics
- #### 2.2.1 Temporal Softmax Loss for Q1 and Q2

- #### 2.2.2 Content Loss and Style Loss for Q4

- #### 2.2.3 GAN Loss for Q5

### 2.3 Results
![RNN Loss](https://github.com/JunStitch/cs231n/blob/master/assignment3/img/rnn_loss.png)

![LSTM Loss](https://github.com/JunStitch/cs231n/blob/master/assignment3/img/lstm_loss.png)

![LSTM Result](https://github.com/JunStitch/cs231n/blob/master/assignment3/img/lstm_caption.png)

![Saliency maps](https://github.com/JunStitch/cs231n/blob/master/assignment3/img/saliency_maps.png)

![Style Transfer](https://github.com/JunStitch/cs231n/blob/master/assignment3/img/style_transfer.png)

![Vanilla Gan](https://github.com/JunStitch/cs231n/blob/master/assignment3/img/vanilla_gan.png)

![ls Gan](https://github.com/JunStitch/cs231n/blob/master/assignment3/img/lsgan.png)

![dc Gan](https://github.com/JunStitch/cs231n/blob/master/assignment3/img/dcgan.png)