from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, classifier, test_idx=None,
                          resolution=0.02):    
    """Plot decision regions of a classifier.

    Parameters
    ----------
    X : array, shape (n_samples, 2)
        Feature Matrix.
    y : array, shape (n_samples,)
        Target vector.
    classifier : object
        Perceptron classifier.
    test_idx : array, shape (n_test_examples,)
        Indices of the test examples.
    resolution : float, optional (default=0.02)
        Resolution of the decision surface.

    """

    # setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')
    # highlight test examples
    if test_idx:
        # plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]
        
        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='none', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o',
                    s=100, label='Test set')

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))



iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

print("Class labeles: ", np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)

print("Misclassified samples: %d" % (y_test != y_pred).sum())
print("Accuracy: %.3f" % accuracy_score(y_test, y_pred))


# Calcula y muestra la precisión del modelo usando la función `accuracy_score` de scikit-learn.
# Esta función compara las etiquetas verdaderas (y_test) con las etiquetas predichas (y_pred).
# Nota: Esta línea es una repetición de la línea 66.
print("Accuracy: %.3f" % accuracy_score(y_test, y_pred))


# Utiliza el método `score` del propio clasificador Perceptron para calcular y mostrar la precisión.
# Este método es una forma conveniente que internamente realiza la predicción sobre X_test_std
#
print("Accuracy: %.3f" % ppn.score(X_test_std, y_test))


X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std,
                      y=y_combined,
                      classifier=ppn,
                      test_idx=range(105, 150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

z = np.arange(-7, 7, 0.1)
sigmoid_z = sigmoid(z)
plt.plot(z, sigmoid_z)
plt.axvline(0.0, color='k')
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\sigma (z)$')
plt.tight_layout()
plt.show()