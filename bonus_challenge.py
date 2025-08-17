# Bonus Challenge: Modified Dataset
import numpy as np
import matplotlib.pyplot as plt

class Perceptron(object):
    """Perceptron classifier with visualization support"""
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data"""
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

# Modified dataset with new book
X_new = np.array([[2, 3], [1, 1], [4, 5], [3, 4]])
y_new = np.array([1, -1, 1, 1])

# Training with modified data
model_new = Perceptron(eta=0.1, n_iter=10, random_state=1)
model_new.fit(X_new, y_new)

print("=== Bonus Challenge Results ===")
print("Prediction for [3,2]:", model_new.predict([3, 2]))
print("Errors per epoch:", model_new.errors_)

# Visualization
plt.figure(figsize=(15, 5))

# Error plot
plt.subplot(1, 2, 1)
plt.plot(range(1, len(model_new.errors_)+1), model_new.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Errors')
plt.title('Learning Progress with Modified Dataset')
plt.grid(True)

# Decision boundary plot
plt.subplot(1, 2, 2)
plt.scatter(X_new[y_new==1][:,0], X_new[y_new==1][:,1], 
            color='blue', marker='o', label='Fiction (+1)')
plt.scatter(X_new[y_new==-1][:,0], X_new[y_new==-1][:,1], 
            color='red', marker='x', label='Non-fiction (-1)')
plt.scatter([3], [2], color='green', marker='*', s=200, label='Test book [3,2]')

# Create decision boundary
x1_min, x1_max = X_new[:,0].min()-1, X_new[:,0].max()+1
x2_min, x2_max = X_new[:,1].min()-1, X_new[:,1].max()+1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                       np.arange(x2_min, x2_max, 0.1))
Z = model_new.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
Z = Z.reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, alpha=0.3, cmap='RdBu')
plt.xlabel('Size')
plt.ylabel('Color')
plt.title('Decision Boundary with New Book [3,4]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('perceptron_results.png')  # Save for GitHub submission
plt.show()

# Experiment with random_state
print("\n=== Random State Experiments ===")
for seed in [1, 42, 100]:
    model_rand = Perceptron(eta=0.1, n_iter=10, random_state=seed)
    model_rand.fit(X_new, y_new)
    print(f"random_state={seed}: Prediction for [3,2] = {model_rand.predict([3, 2])}, "
          f"Final errors = {model_rand.errors_[-3:]}")