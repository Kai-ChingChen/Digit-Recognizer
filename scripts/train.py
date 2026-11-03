import numpy as np
import pandas as pd
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.autograd import Value
import src.functions as F
from src.model import MLP

seed = 42   
rng = np.random.default_rng(seed)

#load data
data  = pd.read_csv(PROJECT_ROOT / "data" / "train.csv")
pixels = data.drop(columns='label').to_numpy(dtype = np.float64)
labels = data['label'].to_numpy(dtype = np.int64)
X = pixels / 255.0  # Normalize pixel values to [0, 1]
print(f"Loaded {len(X):,} \nExamples: X {X.shape}\nLabel {labels.shape}\n")

#data split
indices = rng.permutation(len(X))
cut = int(0.8 * len(X))
train_id, val_id = indices[:cut], indices[cut:]
X_train, y_train = X[train_id], labels[train_id]
X_val, y_val = X[val_id], labels[val_id]
model = MLP(seed)
print(f'Train: {len(X_train)}\nValidation: {len(X_val)}\nParameters: {sum(p.data.size for p in model.parameters())}')

#training
batch_size = 64
epochs = 5
learning_rate = 0.15
history = []

for epoch in range(1, epochs+1):
    order = rng.permutation(len(X_train))
    running_loss = 0.0
    for start in range(0, len(order), batch_size):
        batch_indices = order[start: start+batch_size]
        batch_x = Value(X_train[batch_indices])
        batch_y = y_train[batch_indices]
        for parameter in model.parameters():
            parameter.grad.fill(0.0)

        logits = model(batch_x)
        loss = F.cross_entropy(logits, batch_y)
        running_loss += float(loss.data) * len(batch_indices)
        loss.backward()
        for parameter in model.parameters():
            parameter.data += -learning_rate * parameter.grad

    train_loss = running_loss / len(X_train) #average of all loss
    val_accuracy = F.accuracy(model, X_val, y_val)
    history.append((train_loss, val_accuracy))
    print(f'Epoch {epoch}/{epochs}: train loss {train_loss:.4f}, validation accuracy {val_accuracy: .2%}')

#prediction
test_set = pd.read_csv(PROJECT_ROOT / "data" / "test.csv")
pixels_test = test_set.drop(columns = 'label').to_numpy(dtype = np.float64)
labels_test = test_set['label'].to_numpy(dtype = np.int64)

example_ids = np.random.choice(
    len(pixels_test), size=min(10, len(pixels_test)), replace=False
)

for i in range(len(example_ids)):
    x = example_ids[i]
    print("---------------")
    digit, probabilities = F.predict_digit(model, pixels_test[x].reshape(28, 28))
    print(f'image: {i+1}: predicted {digit}, true label {labels_test[x]}\n')
    print('Probabilities for digits 0-9:', np.round(probabilities, 3))

#save the model
np.savez(
    "my_weights.npz",
    W1=model.W1.data,
    b1=model.b1.data,
    W2=model.W2.data,
    b2=model.b2.data,
    W3=model.W3.data,
    b3=model.b3.data,
    W4=model.W4.data,
    b4=model.b4.data,
)