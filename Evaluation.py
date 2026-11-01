from Model import MLP
import numpy as np
import pandas as pd
import Function as F

test_set = pd.read_csv('data/test.csv')
pixels_test = test_set.drop(columns = 'label').to_numpy(dtype = np.float64)
labels_test = test_set['label'].to_numpy(dtype = np.int64)

example_ids = np.random.choice(
    len(pixels_test), size=min(10, len(pixels_test)), replace=False
)

def predict(model):
    for i in range(len(example_ids)):
        x = example_ids[i]
        print("--------------")
        digit, probabilities = F.predict_digit(model, pixels_test[x].reshape(28, 28))
        print(f'image: {i+1}: predicted {digit}, true label {labels_test[x]}\n')
        print('Probabilities for digits 0-9:', np.round(probabilities, 3))
    print("\n\n\n")
    
#untrained model:
untrained_model = MLP(42)
predict(untrained_model)

#trained model:
trained_model = MLP()

with np.load("model_weights.npz") as weights:
    trained_model.W1.data = weights["W1"].copy()
    trained_model.b1.data = weights["b1"].copy()
    trained_model.W2.data = weights["W2"].copy()
    trained_model.b2.data = weights["b2"].copy()
    trained_model.W3.data = weights["W3"].copy()
    trained_model.b3.data = weights["b3"].copy()

#prediction
predict(trained_model)