from Value import Value
import numpy as np

def cross_entropy(logits, targets):
    if logits.data.ndim != 2 or logits.data.shape[1] != 10:
        raise ValueError("Expected logits with shape (batch, 10)")
    targets = np.asarray(targets, dtype = np.int64)
    batch_size = logits.data.shape[0]
    if targets.shape != (batch_size, ) or not np.all((0 <= targets) & (targets < 10)):
        raise ValueError("Expected one digit(0-9) per row")
    shifted = logits.data - logits.data.max(axis = 1, keepdims = True)
    #softmax
    exp_scores = np.exp(shifted)
    probs = exp_scores/ exp_scores.sum(axis = 1, keepdims = True)
    #loss after simplified algebraically
    losses = np.log(exp_scores.sum(axis = 1)) - shifted[np.arange(batch_size), targets]
    out = Value(losses.mean(), (logits,), 'cross_entropy')
    def _backward():
        #gradient calculation after simplified algebraically
        gradient = probs.copy()
        gradient[np.arange(batch_size), targets] -= 1
        logits.grad += out.grad * gradient/ batch_size
    out._backward = _backward
    return out

def predict_logits(model, images):
    h = np.tanh(images @ model.W1.data + model.b1.data)
    h = np.tanh(h @ model.W2.data + model.b2.data)
    h = np.tanh(h @ model.W3.data + model.b3.data)
    return h @ model.W4.data + model.b4.data

def predict_digit(model, image):
    image = np.asarray(image, dtype = np.float64)
    if image.shape not in [(28, 28), (784,)]:
        raise ValueError("Wrong image format")
    if not np.all((0 <= image) & (image <= 255)):
        raise ValueError("Pixel values must be in [0, 255]")
    scores = predict_logits(model, image.reshape(1, 784)/ 255.0)[0]
    shifted = scores - scores.max()
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    return int(np.argmax(probabilities)), probabilities

def accuracy(model, images, targets, batch_size = 512):
    predictions = np.concatenate([
        predict_logits(model, images[start: start+batch_size]).argmax(axis = 1)
        for start in range(0, len(images), batch_size)
    ])
    return np.mean(predictions == targets)