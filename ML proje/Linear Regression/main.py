import joblib
import numpy as np

with open("multiple_linear_regression_model_19.joblib", "rb") as f:
    model = joblib.load(f)

inputs = np.array([1, 2.0, 2.0, 0.02, 505, 3, 777.0])
inputs = inputs.reshape(1, -1)

prediction = model.predict(inputs)

print(prediction)


