from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt

# 1. Generate some normal data (clustered around 0)
rng = np.random.RandomState(42)
X = 0.3 * rng.randn(100, 2)

# 2. Add some outliers (far from the normal cluster)
outliers = rng.uniform(low=-4, high=4, size=(10, 2))
X = np.vstack([X, outliers])  # Combine normal data and outliers

# 3. Create and fit the Isolation Forest
# 10% of data is expected to be outliers
clf = IsolationForest(contamination=0.1, random_state=42)
clf.fit(X)

# 4. Predict: -1 = outlier, 1 = normal
y_pred = clf.predict(X)

# 5. Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='coolwarm', edgecolors='k')
plt.title("Isolation Forest: Outliers (-1) vs Inliers (1)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()
