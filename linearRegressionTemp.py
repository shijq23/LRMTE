# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import random

x = list(range(0, 20))
y = [1.8 * F + 32 + random.randint(-3, 3) for F in x]


# %%
print(f'X: {x}')
print(f'Y: {y}')


# %%
import matplotlib.pyplot as plt
plt.plot(x,y,'-*r')
plt.show()


# %%
import sklearn
import numpy as np

x = np.array(x).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)

print(f'X: {x}')
print(f'Y: {y}')


# %%
import sklearn.model_selection
xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
print(xTrain.shape)
print(xTrain)


# %%
import sklearn.linear_model
model = sklearn.linear_model.LinearRegression()
model.fit(xTrain, yTrain)
print(f'Cofficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')


# %%
accuracy = model.score(xTest, yTest)
print(f'Accuracy: {accuracy}')


# %%
x = x.reshape(1, -1)[0]
m = model.coef_[0][0]
c = model.intercept_[0]
y = [m*F + c for F in x]

plt.plot(x, y, '-*b')
plt.show()


# %%



