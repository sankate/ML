# Machine Learning

## Table of Contents
1. [Linear Regression Algorithm](#linear-regression-algorithm)
2. [Introduction to Decision Tree](#introduction-to-decision-tree)


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
