# DFANet code

This is an implementation of DFANet. The original authors of the code are https://github.com/huaifeng1993 and https://github.com/jandylin. The code was modified and adapted to be trained in a Container using an RTX 2080 Ti.

## PRE-training process - Backbone (XCeptionA)

1. Download ImageNet in <code>datasets/imagenet</code>. Divide it in training data <code>datasets/imagenet/train</code> and validation data <code>datasets/imagenet/val</code>.
2. Run valprep.sh in <code>datasets/imagenet/val</code>
3. Run <code>python pretrain_xception.py</code>


## Training process - DFANet

1. Download Cityscape's gtFine in <code>datasets/Cityscape/gtFine</code> and leftImg8bit <code>datasets/Cityscape/leftImg8bit</code>
2. Run <code>python train_cityscapes.py datasets/Cityscape -b 2</code>




