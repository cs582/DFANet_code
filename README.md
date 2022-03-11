# DFANet_code

This is an implementation of DFANet. The original authors of the code are https://github.com/huaifeng1993 and https://github.com/jandylin. The code was modified and adapted to be trained in a Container using an RTX 2080 Ti.

## Training process

The first step is to train the backbone. First download ImageNet in <code>datasets/imagenet</code>. Divide it in training data <code>datasets/imagenet/train</code> and validation data <code>datasets/imagenet/val</code>
