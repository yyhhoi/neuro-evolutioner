from sklearn import datasets
import matplotlib.pyplot as plt

# The digits dataset
digits = datasets.load_digits(n_class= 4)

X = digits.images
for i in range(X.shape[0]):
    plt.imshow(X[i], cmap="gray")
    
    plt.show()