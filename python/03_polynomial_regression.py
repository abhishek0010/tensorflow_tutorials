"""
Simple tutorial for using TensorFlow to compute polynomial regression.

Created by Parag K. Mital, Jan. 2016
Modified by Abhishek Kumar, Jan 2020
"""
# %% Imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# %% Let's create some toy data
plt.ion()
n_observations = 100
fig, ax = plt.subplots(1, 1)
X = np.linspace(-3, 3, n_observations)
Y = np.sin(X) + np.random.uniform(-0.5, 0.5, n_observations)
ax.scatter(X, Y)
fig.show()
plt.draw()

# %% Here the values are scaled for shorter range
X_scaled = X/max(X)
Y_scaled = Y/max(Y)

# %% Here the class PolymonialFeatures is used for generating required 
# n degree polymonial for the polymonial equation.
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2) 

# %% The degree of coefficients which fits the curve is determined by
# hit and trial method, or by observing the behaviour of dataset
X_2 = poly.fit_transform(X_scaled.reshape(-1,1))
print(X_2.shape)
print(X_2[0])

poly = PolynomialFeatures(degree=3) 
                                        
X_3 = poly.fit_transform(X_scaled.reshape(-1,1))
print(X_3.shape)
print(X_3[0])

poly = PolynomialFeatures(degree=4)

X_4 = poly.fit_transform(X_scaled.reshape(-1,1))
print(X_4.shape)
print(X_4[0])

# %% A single node model is trained using the scaled polynomial dataset
tf.keras.backend.clear_session()
model = tf.keras.models.Sequential(
    tf.keras.layers.Dense(1, input_shape=[5])
)

model.compile(loss="mean_squared_error",
              optimizer=tf.keras.optimizers.Adam(0.1))

history = model.fit(X_4, Y_scaled, epochs=1000)

plt.plot(history.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.show()

mse = history.history['loss'][-1]
y_hat = model.predict(X_4)

plt.figure(figsize=(12,7))
plt.title('TensorFlow Model')
plt.scatter(X_2[:, 1], Y_scaled, label='Data $(X, y)$')
plt.plot(X_2[:, 1], y_hat, color='red', label='Predicted Line $y = f(X)$',linewidth=4.0)
plt.xlabel('$X$', fontsize=20)
plt.ylabel('$y$', fontsize=20)
plt.text(0,0.70,'MSE = {:.3f}'.format(mse), fontsize=20)
plt.grid(True)
plt.legend(fontsize=20)
plt.show()
