# Classify Species of Birds

## Problem description
The goal of this project is to classify different species of birds as accurately as possible. There is a competition hosted on [Kaggle][kaggle] that provides dataset for training and testing as well as the utility to measure accuracy.

### Dataset
The dataset provided consists of colored images of birds. The training set consists of over 38,000 images with 555 different types of birds, and the test set consists of 10,000 images. The true label of the test set is not provided.

## The approach
The best tool for image classification tasks we know of is deep neural networks. More specifically, convolution neural networks (CNN) are particularly good at image classification due to the fact that pixels closed by are more correlated than pixels far apart. So the high level approach is to train a deep neural network with primarily convolution layers to classify birds.

### Data cleaning
The data provided consists of colored images of birds that are not the same size. For consistency, the image is cropped to a fixed size before feeding into the neural network. The size of the image affects the training speed and accuracy. We found that larger image size provides more accurate classification but slows down the training and evaluation process. `128x128`, `224x224`, and `256x256` are tried during this project.

![image sizes distribution][sizes]

To provide more diverse data, several data augmentation techniques are used, including random horizontal flip, random crop, and color jittering. The data augmentation techniques that are not used due to limited time are random rotation and normalization.

![sample images after training transform][transform]

Since the true labels of test data are hidden, we split 10% randomly off from the provided training set as a validation set. This allows us to better control the training process as we can evaluate the accuracy on validation set after each epoch to gain insight about the test accuracy.

### The network architecture
There are many successful neural network architectures for image classifications. In most cases, they consist of many blocks, and each block consists of convolution layers, max pooling layers, and normalization layers. In the end, the output of the last block is flattened, and a fully connected layer is used to map it to the final classification result, a vector of size *n*, representing the score for each class.

In a fully trained network, layers before the final linear layers behave like feature extractors that progressively extract higher levels of features. Many of these features are independent of the classification task. This enables transfer learning, a technique of reusing part of weights of a trained network for different tasks. Transfer learning also has the benefit of providing better accuracy, as those weights could be trained from a much broader dataset than the task at hand. In general, weights from the previous layers of the existing network can be frozen so that their gradient won’t change during transfer learning. This could result in faster training time but worse accuracy. In this project, we did not freeze the weights. We initialized resnet and efficientnet using pretrained weights, reset their final fully connected layer and train them on this dataset to classify birds.

There are no particular technical reasons for choosing these architectures. We choose them because pretrained weights of these models are easily accessible.

### Training
To train the model, we used the stochastic gradient descent optimizer (SDG) from pytorch. Based on previous experiences with neural networks, we found that a LR of 0.01, momentum of 0.9 and weight decay of 0.001 a good starting point. However, the model begin to plateau (i.e., the losses did not decrease across epochs) as we reached 14 epochs. To improve the model, we used the [cosine annealing][cosine] learning rate scheduler to vary learning rate across the training process. This [scheduler][cosineLR] also reset itself after certain epochs This enables us to train the model longer before it plateaus and therefore getting a better accuracy.

![cosine annealing learning schedule][lr_schedule]

## Results

TODO: loss plot, validation accuracy plot, final test accuracy
Previous submissions test accuracy, configurations

## Discussion

Things to do
More data preprocessing, architecture selection, ensembles.

One thing I would like to try is ensemble models.

## Reference

[pretrained models (TIMM)][TIMM]

[Transfer][transfer1] [Learning][transfer2]

[SGDR: Stochastic Gradient Descent with Warm Restarts][cosine]

[Deep Residual Learning for Image Recognition][resnet]

[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks][efficientnet]

[model][ensemble1] [ensemble][ensemble2] [resources][ensemble3]


[sizes]: imgs/size_dist.png
[transform]: imgs/transform.png
[lr_schedule]: imgs/lr_schedule.png

[kaggle]: https://www.kaggle.com/t/dd340e27d2b745a7bebe35799c0452ba
[TIMM]: https://github.com/huggingface/pytorch-image-models
[transfer1]: https://colab.research.google.com/drive/1EBz4feoaUvz-o_yeMI27LEQBkvrXNc_4?usp=sharing
[transfer2]: https://colab.research.google.com/drive/1kHo8VT-onDxbtS3FM77VImG35h_K_Lav?usp=sharing
[cosine]: https://arxiv.org/abs/1608.03983
[cosineLR]: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html
[resnet]: https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
[efficientnet]: https://proceedings.mlr.press/v97/tan19a/tan19a.pdf
[ensemble1]: https://ensemble-pytorch.readthedocs.io/en/latest/introduction.html
[ensemble2]: https://arxiv.org/pdf/2104.02395.pdf
[ensemble3]: https://towardsdatascience.com/ensembles-the-almost-free-lunch-in-machine-learning-91af7ebe5090