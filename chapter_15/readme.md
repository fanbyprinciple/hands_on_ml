# Autoencoders

Autoencoders are artificial neural networks capable of learningeffiient representations of the input data, called codings without any supervision (i.e training set is unlabelled) 

These typically have lower dimensionality than the input data, so used in dimensionality reduction

These can be used for creating powerful feature detectors, and they can be used for unsupervised pretraining of deepneural networks.

These are capable of generating new data that looks similar to a generative model

These work by copying input to output, we can add nose or limit th esime of internal representation or we can add some noise to the inputs and train the network to recover the original inputs. These contrints prevent the autoencoder fromtrivial copying of inputs to outputs.

## Efficient data representation

Hailstone sequence : even nofollowed by half, off number followed by triple + 1
Autoencoder composed of two parts : encoder (recognition network), and decoder (generative network) that converts the internal representation to the outputs

Autoencoder has the same architecture as the multilayer perceptron, except the number of inputs must be equal to the number of outputs. Outputs are called reconstructions since the autoencoders tries to reconstruct the inputs and loss function contains a reconstruction loss that penalizes the model when reconstructions are different from the inputs

Because internal representationhas a lower dimensionality than th einput data ( it is 2D instead of 3D) the autoencoder is said to be undercomplete. An undercomplete autoencoder cannot trivially copy it inputs.

### simple PCA using autoencoder

If the autoencoder uses only linear activations and the cost function is the Mean Squared Error (MSE) then it can be shown that it ends up performing PCA
![](input_pca.png)

After autoencoder:
![](pca.png)

### reconstruction using auto encoder , stacked autencoder

Also called deep autoencoders, adding more layer hep it develop more complex codings. We should not make autoencoder too powerful otherwise it will create a paerfect copy of input data.

Foe example for mnist it would need input and output equal to 784 units while there can be 3 hidden layers

![](reconstruction.png)

### Tying the weights

We can tie the weights if the autoencoder is perfectly symmetrical, This halves th enimber of wights speeeding the training and limiting the risk of overfitting
![](tying_weights.png)

### Training One Autoencoder at a time

It is faster to train one shallow encoder at a time and then stack all using a single autoencoder

Simplest approach being using different tesnorflow graph:
![](tf_different_graph.png)

Another approach is to use a single graphcontaining the whole stacked autoencoder, plus some extra operations

It has two phases of training
phase 1: in this phase the aimis to produce output as close to input as possible
phase 2: in this pjase the weights one are locked and then the layer 3 output is matched ouput of hidden layer 1. All in all it consists of 5 layers.

Since layer 1 is frozen at phase two we can cache output and then use it when compbining the output of the phases.

![](tf_single_graph.png)

### Visualizing the features 

WE need to look st features autoencoders trained in. The simplest technique is to consider each neuron in every hidden layer and find the training instance that activate it the most. This is especially useful for the top hidden layers since they often capture relatively large features that we can spot them.

However for lower layers thos technique does not work so well.

For each neuron in the first hidden layer we reacreate an iimage where a pixels intensity corresponds to te weight of connection of given neuron.

![](features.png)

Another technique to feed the autoencoder is to feed a random input images, measure the acuracy og teh neuron you are interested in, and then perform backpropagation to tweak the images such that neuron will activat even more
And then by grdient ascent we can tweak the image that the activation increase even more

A simple way to do thsi is however to measure the performance of the classifier

### Unsupervised Pretraining Using Stacked Autoencoders

WE can always reuse layers, like transfer learning.
Similarly if we have a large dataset but most of it unlabelled , you can first train a stacked autoencoder using all the data then reuse the lower layers to create a neural network for your actual task and train it using the labeled data. 

For example we can make a stack autencoder, when training the classifier, if we dont have labelled data then we freeze the pretrained layer
This is good because the large unlabelled dataset is often chaep, and labelling them is time consuming

In implementation we just reuse the encoder layer ti create a new neural network

### Denoising autoencoders

We can make output even bigger than input it is called an overcomplete autoencoder. Denoising is a way to force autoencoders to add noise in inputs , traiing it to recover the original noise free inputs. This prevents from just copy pasting and having to find patterns for autoencoder.

We can also use dropout along with the noise to make the autoencoder.
![](denoising.png)


### Sparse Autoencoders

If we reduce the active number of neurons then we force the neurons to be a cumulative of multiple neurons, We need to find the sparsity of each layer at coding iteration.

Once we have the mean activation per neuron,we want to penalize the neurons that are too active by adding sparsity loss to the cost function, activation should be more than the targer sparcity, in order to do it we use the KL divergence or Kullback-Leibler divergence, which has much stronger gradient.

![](sparsity_graph.png)

once we get the saprsity liss for each neuron in the coding layer, we just sum up theselosses and add the result to the cost function. In order to control the relative importance of sparsity loss and reconstruction loss we can multiply sparsity loss by a sparsity weight hyperparameter. 

If weight is too high then model will stick closely to the target sparsity, it may not reconstruct the inputs properly, making the model useless. If its too low, the model will mostly ignore and not learn any useful features

![](sparse_output.png)

## Variational Autoencoders

These are probabilistic autoencoders : outputs are partially determined by chance even after training (as opposed to denoising autoencoders, which use randomness only during training.)

Most importantly they are generative autoencoders meaning that htey can generate new instances that look loke they were sampled from training set.

But they are easier to train the sampling process similar to RBMs. Restricted boltzman machines , in which we need to wait for the network to stabilize into a thermal equilibrium before we can sample a new instance.

we actaully add gaussian noise in beterrm
the mean coding _mu_ and the standard devaition _sigma_ are sampled with gaussian noise

As we can see although the inputs may have a very convoluted distribution a variational autoencoder tends to produce codings from gaissian noise. One great consequence is that after a varaitional autoencoder, you can easily generate a new instance

Lets lookat cost function. It has two parts the first is the usual reconstruction loss that pushes the autoencoder to reporduce its inputs , the second is the latent lossthat pushes the autoencode to have codings as if they were sampled a simple gaussian distribution for which we will use the KL divergence between the targets and the actual distribtion of the codings, The math is a bit more complex. 

![](variational_autoencoder.png)

The encoding and decoding through a variational autoencoder:

![](encode_decode.png)

The interpolation of digits

![](interpolation.png)

## other examples of autoencoders

1. contractive autoencoder(CAE)
2. stacked convolutional autoencoder - for extracting visual features
3. Generative stochastic network - genralisation of denoising autoencoders
4. Winner take all autoencoder - naturally leads to sparse codings
5. Adverserial autoencoder -  network trained to reproduce its intpust with an adverserial network

## Exercises 

1. Autoencoder are used for 
    1. feature extraction
    2. unsupervised training
    3. dimensionally reduction
    4. generative models
    5. anomaly detection (an autoencoder is generally bad at reconstructing outliers)
    
2. If we have plenty of unlabelleddata then we canfirst train a deep autoncoder on the full dataset (labeled + unalbelled) then reuse the lower half og classifier (that is uptil the codinglayer and then train using the labelled data. If we have labeled data, we want to freeze the reused layers when training the classifier

3. If autoencoder perfectly constructs its inputs then it does not mean it is a good autoencoder, perhaps it is simply an overcompletet autoencoder that learned to copy its inputs layers to the coding layer then to outputs. In fact even coding layer contained a single neuron it would be possible for a very deep autoencoder to learn to map each training instance to a different coding. 

Perfect autoencoders dont learn anyuseful features
however bad reconstructions do mean bad autoencoder
To evaluate the performance ofan autoencoder one option is tomeasure its recontruction loss, calculate MSE , mean square of outputs minus the inputs, if you are using anautoencoder for a particular thing, then evaluate acccordingly, eg classifier performance.

4. An undercompleteautoencoder isonewhoe coding layer is smaller tan the input and output layers. If its larger then its overcomplete

5. To tie weight of decoder and encoder we make decoder weights a transpose of encoder weights, thia reduces the number of parameters by half

6. TO visualise the features by the lower layer of a stacked autoencoder , a commontechniquw i to simply plot the weights of each neuronby reshaping each each weight vector to the size of an input image (eg MNIST, reshaping a weight vector of shape [784] to [28,28].To visualize the features learned by higher layers one technique is to display the traiing instnace that most activate each neuron.

7. a generative model is a model capable of randomly generating outputs that resemble the trsining instances , for exmple once trained successfully onthe mnsit dataseta generative model can be used to randomly generate realistic image of the digits, the distribution being the same as the input distribution.An example of generative autoencoder is the variational autoencoder.

# to do

8. Building a CIFAR 10 image classifier using denoising autoencoder

9. semantic hashing: we can retireive documents that are similar 

10. Using cifardataset to construct images




