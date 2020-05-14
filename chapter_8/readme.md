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

Many dimensionality reduction algorithms work by modelling the manifold on which training instances lie this is called manifold learning 

it relies on manifold assumption also called manifold hypothesis which holds modt real world high dimensionality datasets lie lose to a much lower dimensional manifold. 

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

Finding the principle componenet is done through singular value decompositions

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

It stochastically builds an approximation on given number of features. It is quick.

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

for each training instance x(i) the algorithm identifies its k closest neighbors (in the preceding code k=10) then tries to recontruct. x as a linear function of there neighbhors more specifically it finds weights w(i,j) such that the squared distance between x and sum(w(i,j)x(j)) is as small as possible. assuming w(i,j) =0 if x is not the k closest neigbhor of x(i)

THis first step is a contraint optimisation problem with weights W.the second constraint simply normalises the weights for each training instances x(i).

computational complexity:
for finding k nearest neighbors O(m log(m)nlog(k))
for optimising wieghts O(mnk^3)
for recontsruction of lowe dimesnional represenstation O(dm^2)

![](lle.png)


### MDS, Isomap and t-SNE

MDS :  multi dimensional Scaling - reduces dimensionality whilepreserving disatnace between instances.

Isomap : creates a graph by connecting each instances to its nearest neighbors then reduces dimensionality while trying to preserve geodesic distances between the instances, geodesic distance between two nodes is a graph is the number of nodes on the shortest path berween these nodes.

t-SNE : reduces dimensionality while tryinig to keep similar instances close and dissimilar instances apart used for visualisation

Linear Discriminant Analyssis :  Is actualy a classificiation algogrithm but during training it learns the most discriminative axes between the classes and these axes can then be used to define a hyperplane onto which to project the data. The benefit being that the projection will make the classes as fara apart as possible, so LDA is a good technique to reduce dimensionality before applying something like the svm classifier.

![](mit.png)

# Clustering and classification

On Iris dataset typical representation of cluster and classification
![](cluster_vs_classify.png)

We can do cluster classfiication usign gaussian mixture
![](gaussian_mixture.png)

### K means
We can do clustering through K meansalgorithm

![](kmeans.png)

Hard clustering -  making predictions at an arbitary distance form centroid
soft clustering - using euclidean distance to make clusters

Algorithm:

first initialise k centroids randomly: k distinct instances are chosen randomly from dataset and the centroids are placed at their locations

Repeat until convergence
assign each instance to the closest centroid
update the centroids to be the mean of the instances that are assigned to them

![](kmeans_action.png)

Kmeans is optimised version. Original K means can be obtained by set init="random", n_init=1 and algorithm="full"

K means variability
![](kmeans_variability.png)

### K means ++

Instead of inititalising the centroids randomly we can inititlise using algorithm given by David Arthur and Sergei Vassilvitskii

Take one centroid c1
Take a new center c(i) choosing instance x(i) with probability D(x(i)^2/ sum(D(x(j)^2)
where D(x(i)) is the distance between instance x(i) and the closest centroid that was already chosen.

repeat till allcentroids are chosen
Rest is just reguar Kmeans. we just have to select k-means++ , whic is by default

![](kmeans_plusplus.png)

### K means minibatch

![](kmeans_minibatch.png)


### Elbow diagram

increasing the value of k increases inertia but doen't help in classification

![](bad_classification.png)

we can find the best value of k by elbow diagram

![](elbow_diagram.png)

choosing the correct value of k :
![](correct_k.png)

### silhouette diagram

![](silhouette_vs_k.png)

### silhoette analysis diagram

It is an informative visualisaiton where we plot every instance's silhoette coefficient, sorted by cluster they are assigned to and the value of coefficient.

This is called silhoette diagram:
![](silhouette_diagram.png)

limits of kmean
![](limits_of_kmeans.png)

### K means for image segmentation

so each of the 425400 instances is replaced by the centroid value of clusters

![](flower.png)



### K means for preprocessing
we will try and improve accuracy by using preprocessing by kmeans as a
preprocessing step. We will create a pipeline that will first cluster the training set into 50
clusters and replace the images with their distances to 50 clusters then apply
logistic regression

![](preprocessing.png)

### K means for semi supervised learning

It means we have plenty of unlabelled data and very few labelled instances

![](semi_supervised.png)


### DBSCAN

Density based spatial clustering of applications with Noise

DBSCAN will start by dividing the data into n dimensions. After DBSCAN has done so, it will start at a random point (in this case lets assume it was one of the red points), and it will count how many other points are nearby. 
![](dcgan1.png)

eps indicates howclose must the instnaced be to be taken as part of the same cluster.
![](dbscan.png)

using knn
![](knn.png)

### spectral clustering

https://towardsdatascience.com/spectral-clustering-aba2640c0d5b

Key concepts:

Eigen vectors
We can think of the matrix A as a function which maps vectors to new vectors. Most vectors will end up somewhere completely different when A is applied to them, but eigenvectors only change in magnitude. If you drew a line through the origin and the eigenvector, then after the mapping, the eigenvector would still land on the line. The amount which the vector is scaled along the line depends on λ.

Graph
A connected component is a maximal subgraph of nodes which all have paths to the rest of the nodes in the subgraph.

adjacency matrix: matrix containing which nodes are connected to which

degree matrix: how many edges are connected to a node is the degree. a matrix of all these degrees is called degree matrix.

Laplacian matrix is a representation of graph given by D -A

No. of zero eigen values represent the connected componenet of the graph

The first eigenvalue is 0 because we only have one connected component (the whole graph is connected). The corresponding eigenvector will always have constant values (in this example all the values are close to 0.32).

![](spectral1.png)

The first nonzero eigenvalue is called the spectral gap. The spectral gap gives us some notion of the density of the graph. If this graph was densely connected (all pairs of the 10 nodes had an edge), then the spectral gap would be 10.

![](spectral2.png)

The second eigenvalue is called the Fiedler value, and the corresponding vector is the Fiedler vector. The Fiedler value approximates the minimum graph cut needed to separate the graph into two connected components. Recall, that if our graph was already two connected components, then the Fiedler value would be 0. Each value in the Fiedler vector gives us information about which side of the cut that node belongs.

![](spectral3.png)

Essentially in spectral clustering all node values are points in graph. And to make cluster we just look at the lowest value eigen vector, if the no. of cluster are 4 then we look at first four eigen vectors


![](spectral_clustering.png)

### Agglomerative clustering

The agglomerative clustering is the most common type of hierarchical clustering used to group objects in clusters based on their similarity. It’s also known as AGNES (Agglomerative Nesting). The algorithm starts by treating each object as a singleton cluster. Next, pairs of clusters are successively merged until all clusters have been merged into one big cluster containing all objects. The result is a tree-based representation of the objects, named dendrogram.

Works in bottom up manner.

### gaussian mixture

A generative clustering algorithm, it can be used to create X and y. 
After fitting

![](gaussian_mixture1.png)

gaussian full vs tied covariance:
![](gaussian_compare1.png)

gaussian spherical vs diag covariance:
![](gaussian_compare2.png)

AIC and BIC used to measure effectiveness of gaussian mixture

![](bic_bestk.png)


### Variational Bayesian Gaussian Mixtures

Rather than optimally searching for clusters it is possible to use instead bayesianGaussianMixture class which iscapable of giving weights equal or close to zero to unnecessary clusters. 

We have to set the number of components to a value that is greater than optimal clusters and algorithm will remove unecessary clusters automatically.

![](bgm.png)


Comparing different values of weight concentration prior
![](wcp.png)

Bayesian gaussian Mixture on moon dataset
![](moon_bgm.png)

### Liklihood estimator

![](liklihood_estimator.png)








