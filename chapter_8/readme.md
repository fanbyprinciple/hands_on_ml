# Dimensionality reduction

Is a way to reduce the complexity if features, known as the curse of dimensionality.

Dimensionally is also very useful for data visualisation.
It helps us gain some important insights.
High dimesnional hypercubes are very sparse.

### Projection to remove dimensionality

In most real world problems the training instances are not spread uniformly accross all dimensions.

projection however doesn'twork when the model is omsthing like a swiss roll.

![](swiss_roll.png)

### manifold learning

A 2d manifold is a 2D shape that can be bent and twisted in ahigher dimensional space. A d dimensional manifold is a part in a higher-dimensional space. More generally a d dimensional manifold is a part of an n-dimensional space (d<n) that locally resembles a d hyperplane.

example of 3D dataset lying close to a 2D subspace:
![](3d_dataset.png)

Converting it into projection in 2D plane:
![](3d_dataset_projection.png)

Many dimensionality reduction algorithms work by modelling the manifold on which training instances lie this is called manifold learning it relies on manifold assumption also called manifold hypothesis which holds modt real world high dimensionality datasets lie lose to a much lower dimensional manifold. 

in mnist the degree of freedom to create an digit image is very lower than degree of freedom to create any image of your choice. these contraints squeeze the datset into a lower dimensional manifold.

Manifold assumption is often accompanied by another implicit assumption. that the task at hand will be simpler if expressed in the lower dimensional space of the manifold.

However its not always true that dimensionality reduction will help in easy creation of decision boundary. Evenif it speeds up training time.

![](swiss_roll_projection.png)

## PCA

Principle component analysis(PCA) identifies the hyperplane closes to the data, and then projjects data onto it.

On swiss roll of arbitary positive negative
![](swiss_1.png)

Changing postive negative and impact on swiss roll
![](swiss_2.png)

### preserving the variance

before actual projection, we need to choose the right hyperplane. 
we choose the axis that preserves most amount of variance.Or the axis that minimises the mean squared distance between the original dataset and its projection onto that axis. This is the idea behind PCA.


PCA variance plot
![](pca_variance.png)

### principal component

A unit vector that defines the ith axis is called ith principal component (PC). 

Finding the principle componenet is done through sinular value decompositions

Decomposing the training set matrix X into dot product of three matrices U . Sigma . V.T where V the contains the principle component we are looking for

The direction of PC is not stable it changes direction when we perturb the data and run PCA again. However the plane defined will be the same.

PCA assumes that the datset is centered around the origin. Scikit-Learn's PCA classes take care of centring the data for yourself. HOwever if you implement PCA yourself don't forget to center the data first.


### projecting to d dimensions

Selecting the hyperplane ensures that the projection will preserve as much variation as possible. 

Effect of parameters on variance:
![](effect_of_params_on_variance.png)

Projection simple means computing the dot product of the traiing set matrix X by the matrix Wd defined as the matrix containing the first d principal components.

### PCA for compression

When we apply PCa over mnist we find that each feature has just 150 features instead of 784, original features.

We can also decompress by applying inverse PCA projection transform. The original data is not given back compelety though. 5% is dropped. 

The mean square distance betweem the original data and the reconstructed data us called reconstruction error.

mnist compression:
![](mnist_compression.png)

### Incremental PCA

One problem with normal PCA is that it requires the whole training set to fit in memory for SVD algorithm to run.

In incremental PCA you can split the training set into mini batches and feed the Inceremental PCA algorithm one mini batch at the time.

![](incremental_pca.png)

### randomised PCA

his stochastically bulds an approximation on give number of features.

PCA vs Randomised on number of samples
![](pcavsrpca1.png)

PCA vs Randomized on number of features
![](pcavsrpca2.png)

### Kernal PCA

Kernal trick is a mathematical technique that maps instances to a very high dimensional space called feature space. This enables us to do nolinear classification and regression. 

A linear decision boundary in high_dimesnional space  corresponds to complex nonlienar decision buondary in original space.

It turns out that same trick can be applied to PCA, makingit possible to perform complex non linear projections for dimensionality reduction. This is called Kernal PCA.

differnet kernal PCA transformation based on kernel:
![](kernel.png)

kPCA is good at preserving clusters of instances after projection. and even unrolls datasets.

### Local Linear Embedding

It is another very powerful nonlinear dimesnionality reduction technique. It is a manifold learning technique that does not rely on projection . 

LLE works by first measuring how each training instance linearly relates to its closest neighbhors and then looking for low - dimensional representation of the thre training set where these local relationships  are best preserved.

![](lle.png)


### MDS, Isomap and t-SNE

![](mit.png)




