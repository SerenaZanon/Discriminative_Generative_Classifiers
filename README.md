# Assignment: Discriminative_Generative_Classifiers
# University course: FOUNDATIONS OF ARTIFICIAL INTELLIGENCE

Write a handwritten digit classifier for the MNIST database. These are composed of 70000 28x28 pixel gray-scale images of handwritten digits divided into 60000 training set and 10000 test set.

In Python, you can automatically fetch the dataset from the net and load it using the following code:

```python
from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
y = y.astype(int)
X = X/255.
```

This will result in 784-dimensional feature vectors (28$\times$28) of values between 0 (white) and 1 (black).

Train the following classifiers on the dataset:
- SVM using linear, polynomial of degree 2, and RBF kernels;
- Random forests;
- Naive Bayes classifier where each pixel is distributed according to a Beta distribution of parameters $\alpha, \beta$:
  $d(x; \alpha, \beta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)\Gamma(\beta)} \alpha^{\alpha - 1} (1 - x)^{\beta - 1}$
- k-NN.

You can use scikit-learn or any other library for SVM and random forests, but you must implement the Naive Bayes and k-NN classifiers yourself.

Use 10-way cross validation to optimize the parameters for each classifier.

Provide the code, the models on the training set, and the respective performances in testing and in 10-way cross validation.

Explain the differences between the models, both in terms of classification performance, and in terms of computational requirements (timings) in training and in prediction.

For a discussion about maximum likelihood for the parameters of a beta distribution you can look here. However, for this assignment the estimators obtained with the moments approach will be fine:

$\alpha = \mathbb{E}[X], \quad \beta = K(1 - \mathbb{E}[X])$
with
$K = \frac{\mathbb{E}[X](1 - \mathbb{E}[X])}{\text{Var}(X)} - 1$

Note: $\alpha/(\alpha + \beta)$ is the mean of the beta distribution. If you compute the mean for each of the 784 models and reshape them into 28x28 images you can have a visual indication of what the model is learning.
