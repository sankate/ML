# Machine Learning

## Table of Contents
1. [Linear Regression Algorithm](#linear-regression-algorithm)
2. [Introduction to Decision Tree](#introduction-to-decision-tree)
3. [Accuracy, Recall, Precision and F-1 score](#classification-metrics-accuracy-precision-recall-and-f1-score)


## Linear Regression Algorithm

**Linear regression** is one of the simplest and most widely used algorithms in machine learning. It models the relationship between a dependent variable (target) and one or more independent variables (features) by fitting a linear equation to observed data. The equation for a simple linear regression (with one feature) is:

$$y = \beta_0 + \beta_1 x $$

- $`y `$: Dependent variable (target)
- $`x `$: Independent variable (feature)
- $` \beta_0 `$: Intercept of the regression line
- $` \beta_1 `$: Slope of the regression line

For multiple linear regression (with multiple features), the equation extends to:

$$ y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n $$

Here, $` x_1, x_2, \ldots, x_n `$ are the independent variables and $` \beta_1, \beta_2, \ldots, \beta_n `$ are the coefficients corresponding to these variables.

### How to Assess the Performance of a Regression Model

The performance of a regression model is typically assessed using the following metrics:

1. **Mean Absolute Error (MAE)**:
   $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
   - Measures the average magnitude of the errors in a set of predictions, without considering their direction.

2. **Mean Squared Error (MSE)**:
   $$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$
   - Measures the average of the squares of the errors. It gives higher weight to larger errors, making it more sensitive to outliers than MAE.

3. **Root Mean Squared Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\text{MSE}}$$
   - Provides an error measure in the same units as the target variable.

4. **R-squared (\( R^2 \))**:
   $`R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}`$
   

   - Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables. Values range from 0 to 1, with 1 indicating a perfect fit.

### Pros and Cons of Linear Regression

**Pros**:
1. **Simplicity**:
   - Linear regression is easy to understand and implement.
   
2. **Interpretability**:
   - The model's coefficients provide clear insights into the relationship between the target and the features.

3. **Efficiency**:
   - Linear regression can be computed efficiently, even with large datasets.

4. **Less Complex**:
   - It requires fewer computational resources compared to more complex models.

**Cons**:
1. **Linearity Assumption**:
   - Assumes a linear relationship between the dependent and independent variables, which might not always be the case.

2. **Sensitivity to Outliers**:
   - Linear regression is sensitive to outliers, which can distort the model significantly.

3. **Multicollinearity**:
   - The presence of highly correlated features can affect the model's performance.

4. **Limited to Simple Relationships**:
   - Cannot capture complex, non-linear relationships.

### Conclusion

Linear regression is a foundational machine learning algorithm that is widely used due to its simplicity and interpretability. It is best suited for problems where the relationship between the variables is approximately linear. For more complex relationships, other models might be more appropriate.

### Sources
- [Wikipedia on Linear Regression](https://en.wikipedia.org/wiki/Linear_regression)
- [Scikit-Learn Documentation on Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
- [Towards Data Science - Linear Regression in Python](https://towardsdatascience.com/linear-regression-in-python-9a1f5f000606)


## Introduction to Decision Tree

**Decision Trees** are a type of supervised learning algorithm that can be used for both classification and regression tasks. They work by recursively splitting the data into subsets based on the value of input features, creating a tree-like model of decisions. Each internal node represents a "decision" or "test" on an attribute, each branch represents the outcome of the test, and each leaf node represents a class label (for classification) or a continuous value (for regression).

### How to Build a Decision Tree

1. **Select the Best Attribute**:
   - Choose the attribute that best splits the data using a criterion like Information Gain, Gini Index, or Mean Squared Error (for regression).

2. **Split the Dataset**:
   - Split the dataset into subsets based on the selected attribute's values.

3. **Create a Node**:
   - Create a decision node that branches out to these subsets.

4. **Repeat Recursively**:
   - Apply the above steps recursively to each subset until one of the stopping conditions is met (e.g., maximum depth, minimum number of samples per leaf, or no more improvement).

### Methods of Pruning a Decision Tree

Pruning is the process of removing parts of the tree that do not provide additional power in predicting target variables, which helps to reduce overfitting.

1. **Pre-pruning (Early Stopping)**:
   - Stop growing the tree before it becomes too complex. This can be controlled by parameters like maximum depth, minimum samples per leaf, and minimum samples per split.

2. **Post-pruning**:
   - Allow the tree to grow fully and then remove nodes that provide little to no contribution. This can be done by:
     - **Cost Complexity Pruning (CCP)**: Prune nodes by minimizing a cost complexity measure that balances the size of the tree and its performance on training data.
     - **Reduced Error Pruning**: Remove nodes only if the removal improves the performance on a validation set.

### Different Impurity Measures

Impurity measures are used to decide the best way to split the data at each node.

1. **Gini Index**:
   $$Gini(D) = 1 - \sum_{i=1}^{n} p_i^2$$
   - Measures the impurity of a node. A node is pure if all samples belong to a single class.

2. **Entropy**:
   $$Entropy(D) = - \sum_{i=1}^{n} p_i \log_2(p_i)$$
   - Measures the disorder or uncertainty in the data. Lower entropy indicates a purer node.

3. **Information Gain**:
   $$IG(D, A) = Entropy(D) - \sum_{v \in Values(A)} \frac{|D_v|}{|D|} Entropy(D_v)$$
   - Measures the reduction in entropy or impurity after a dataset is split on an attribute.

4. **Mean Squared Error (MSE)** (for regression trees):
   $$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
   - Measures the average of the squares of the errors between predicted and actual values.

### Regression Trees

**Regression Trees** are decision trees used for predicting continuous values. Instead of using classification metrics like Gini Index or Entropy, regression trees typically use Mean Squared Error (MSE) to split nodes. The objective is to minimize the variance of the target variable in the leaf nodes.

### Pros and Cons of Using Decision Trees

**Pros**:
1. **Easy to Understand and Interpret**:
   - Decision trees are simple to visualize and interpret, making them accessible to non-experts.

2. **No Need for Feature Scaling**:
   - They do not require normalization or scaling of data.

3. **Handles Both Numerical and Categorical Data**:
   - Capable of handling different types of input data.

4. **Non-parametric and Non-linear**:
   - Can capture non-linear relationships between features and the target.

**Cons**:
1. **Overfitting**:
   - Decision trees can easily overfit the training data, especially if they are deep.

2. **Instability**:
   - Small variations in the data can result in a completely different tree structure.

3. **Biased to Dominant Classes**:
   - If some classes dominate, the tree can become biased towards those classes.

4. **Not Great for Extrapolation**:
   - Poor performance when predicting values outside the range of the training data (for regression).

### Sources
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/modules/tree.html)
- [Wikipedia on Decision Tree Learning](https://en.wikipedia.org/wiki/Decision_tree_learning)
- [Towards Data Science - Decision Trees](https://towardsdatascience.com/decision-trees-in-machine-learning-641b9c4e8052)


## Classification Metrics: Accuracy, Precision, Recall, and F1-score

In the context of machine learning, classification metrics are used to evaluate the performance of classification models. Here's a detailed explanation of four commonly used metrics: Accuracy, Precision, Recall, and F1-score.
Confusion in classifying the data can be shown by a matrix called the Confusion Matrix. From the confusion matrix, we can obtain different measures like Accuracy, Precision, Recall, and F1 scores.
![Confusion \matrix.png "Confusion matrix"](https://github.com/sankate/ML/blob/main/Confusion%20matrix.png)

### 1. Accuracy

**Definition**: Accuracy is the ratio of correctly predicted instances to the total instances. It is a straightforward metric to understand and calculate but can be misleading if the data is imbalanced.

$$ \text{Accuracy} = \frac{\text{True Positives (TP)} + \text{True Negatives (TN)}}{\text{Total Population}}$$

**Usage**: Accuracy is useful when the classes are balanced, meaning there are roughly equal numbers of instances in each class.

**Example**: If a model correctly predicts 90 out of 100 instances, the accuracy is 90%.

### 2. Precision

**Definition**: Precision is the ratio of correctly predicted positive observations to the total predicted positives. It measures the accuracy of the positive predictions.

$$\text{Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}$$

**Usage**: Precision is particularly useful in scenarios where the cost of false positives is high. For instance, in email spam detection, high precision means fewer legitimate emails are incorrectly labeled as spam.

**Example**: If a model identifies 70 true positives out of 100 predicted positives, the precision is 70%.

### 3. Recall (Sensitivity or True Positive Rate)

**Definition**: Recall is the ratio of correctly predicted positive observations to the all observations in the actual class. It measures the ability of the model to find all relevant cases.

$$\text{Recall} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}$$

**Usage**: Recall is important when the cost of false negatives is high. For instance, in medical diagnosis, high recall ensures that most of the actual positive cases are identified.

**Example**: If a model identifies 70 true positives out of 80 actual positives, the recall is 87.5%.

### 4. F1-score

**Definition**: The F1-score is the harmonic mean of precision and recall. It provides a single metric that balances both precision and recall, making it useful when you need a balance between the two.

$$\text{F1-score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Usage**: The F1-score is useful when you need to balance precision and recall, especially in situations with imbalanced datasets.

**Example**: If the precision is 0.7 and the recall is 0.8, the F1-score is:

$$\text{F1-score} = 2 \times \frac{0.7 \times 0.8}{0.7 + 0.8} \approx 0.75$$

### Practical Considerations

- **Imbalanced Data**: When dealing with imbalanced data, accuracy may not be the best metric as it can be misleading. Metrics like precision, recall, and the F1-score can provide more insight into the model's performance.
- **Choice of Metric**: The choice of metric depends on the specific problem and the costs associated with false positives and false negatives. For instance, in fraud detection, false negatives (missing a fraud case) might be more costly than false positives.

### Sources
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Wikipedia on Precision and Recall](https://en.wikipedia.org/wiki/Precision_and_recall)
- [Towards Data Science - Evaluation Metrics](https://towardsdatascience.com/evaluation-metrics-for-classification-4256f54e18af)

These metrics provide different insights into the performance of a classification model, and the best choice often depends on the specific context and goals of the analysis.



## K-means Clustering
**1. What are some applications of clustering in real-world scenarios?**

Common applications of clustering include

Customer Segmentation
Document Clustering
Image Segmentation
Recommendation Engines
 

**2. What is K-means clustering?**

K-means is a centroid-based algorithm, or a distance-based algorithm, where we calculate the distances to assign a point to a cluster. In K-Means, each cluster is associated with a centroid.

 

**3. What are some good things about K-means clustering?**

It is very smooth in terms of interpretation and resolution.
For a large number of variables present in the dataset, K-means operates quicker than hierarchical clustering.
While redetermining the cluster center, an instance can modify the cluster.
K-means reforms compact clusters.
It can work on unlabeled numerical data.
 

**4. What are the limitations of K-means clustering?**

Sometimes, it is quite tough to figure out the appropriate number of clusters, or the value of k.
The output is highly influenced by the original input, for example, the number of clusters.
It gets affected by the presence of outliers in the data set.
In some cases, clusters show complex spatial views, then executing clustering is not a good choice.
 

**5. Is there any metric to compare clustering results?**

You can compare clustering results by checking silhouette scores and by doing cluster profiling. Besides this, you should also validate the clustering results by consulting with a domain expert to see if the cluster profiles make sense or not.

 

**6. For K-means if there is a y-dependent variable, do we remove it before trying to group customers?**

Yes, if you have a dependent variable in your dataset, you should remove that before applying clustering algorithms to your dataset.

 

**7. How do we select the optimal number of clusters from the Elbow curve?**

Choosing the optimal number of clusters is a fairly subjective matter, and the best method to identify the optimum number of clusters is to use a combination of metrics and domain expertise. The Elbow curve is one of the most common ways of finding the right number of clusters for K-Means clustering if we don't have domain expertise. The elbow curve is plotted between the number of clusters on the X-axis and WCSS (within the cluster sum of squares) on the Y-axis.

The elbow method uses the WCSS to choose an ideal value of k based on the distance between the data points and their assigned clusters. WCSS is the sum of the squared distance between each point and the centroid in a cluster. We would choose a value of k where the WCSS begins to flatten out, and we see an inflection point.

![K-mean.png "Elbow curve"](https://github.com/sankate/ML/blob/main/K-mean.png)

The graph above shows that k = 4 is an appropriate number of clusters to choose from, with an obvious elbow at that number. At K=4, the graph shows a significant fall in WCSS. As a result, 4 is the best K-value.
