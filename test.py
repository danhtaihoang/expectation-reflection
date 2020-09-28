from expectation_reflection import classification

x = 2
y = 10

model = classification.model(max_iter = 10, regu = 0.01)

model.fit(x,y)

y_pred = model.predict(4)
print(y_pred)