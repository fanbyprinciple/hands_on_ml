CNN - convolution neural nertwork are also used for voice recognition and natural language processing (NLP). LeNet was used or handwriting recognition.
If we uses simple deep network for recognising pictures it wont work unless the dataset is mnist. 

The convolutionalneurak network : The convolutional layer neurons in the first convolutional layer are not connectedto evey single image in the output image but only to the pixels in their receptive fields. This architecture helps the network to concentrate  on low level features in the first hidden lyer, then assemblethem into  a higher  level feature

A convolution is a mathematical operation that slides one functionover another and measures the integral of thwir pointwise multiplication,It has deep connections with fourier transform and laplace transform.

Each layer is represented in 2D.

A neuron located in row i, column j of given layer is connected to the output of rows i + fh -1 columns j to j + fw - 1 where fh and fw are the height and width of receptive field. In order for a layer to have same shape then we must use padding. Also called zero padding. 

It is also possible toconnect to large input layer by spacing our receotive fields, the distance between convolutional network is called a stride.

### Filters

A neuron's weights can be represented as a samll image of size of receptive filed. There can be filters , fitrst one is a black square with vertical whiteline in the middle ( 7 * 7 matrix with 0s with onlyonw in middle), the subsequent layer will igonre everytnif but tbe 1s


### Stacking multiple Feature maps

A cnn is actually 3D this is handled though conv nets, within one feature map the neurons share parameter this heos inredu. This helps in dramatically reducing the number of parameters

### Tensorflow implementation

In tensorflow each input image is typically represented as  3D tensor of shape (height, width, channels) a mini batch as 4D tensor of size , height , width and channels The weights represented by fh, fw, fn , fn' The bias as [fn]

![](temple.png)

when filter has all zeros:
![](temple_in_night.png)

###  memory requirement

CNNs require a huge amount of RAM especially ditong train, because reverse pass of backpropagation requires all the intermediate values computed during forward pass

During inference i.e when making predictions for a new instance the RAM occupied by one layer can be released as soon as the next layer has been computed, so we need as much ram as required by two consecutive layers

rducing mini batch size can reduce memory needed, also dimensionality of strides  or removing some layer, we can also use 16 bit floats instead of 32 bit floats, we can distribute the CNN accross multiple devices