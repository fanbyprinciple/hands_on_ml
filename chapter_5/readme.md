# Support Vector Machines

## Linear SVM Classification

Two classes can be seperated linearly, they are linearly seperable. SVM comes up with a line that seperates the two classes and stays away from the closest training instances. Also caleed large margin classification.
![](large_margin_classfication.png)

They are dependent on training instances that are on the edge and don
t generally get effected by adding more training instances.

They are sensitive to feature scale.
![](feature_scale.png)

They are also sensitiveto outliers.
![](outliers_scale.png)

Support vectors are the vectors that are used to build the hyperplane or decision boundary.

Hard margin classification : all instances should be at either side of the line. Issues : only works if data is linearly seperable, and it is quite sensitive to outliers. To avoid er use soft margin classificaiton.

Soft margin classification : It allows for margin violations and gives an approximately good model deviding the datasets. 
Using higher C value leads to less violation but less wider margins as well.
using lower C value leads to more violation but wider margin. This will generalise better. Best way to regularise it by reducing c.

Large margin vs small margin wrong.
![](large_small.png)
