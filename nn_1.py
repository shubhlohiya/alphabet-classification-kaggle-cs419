import numpy as np
import pandas as pd
# import tqdm

np.random.seed(42)
epsilon = 1e-7

def ReLU(z):  # ReLU activation function
    return np.maximum(0, z)

def ReLUPrime(z):  # derivative of ReLU activation function
    return 1 * (z > 0)

def softmax(logits):
    """
    Implement the softmax function
    Inputs:
    - logits : A numpy-array of shape (n * number_of_classes )
    Returns:
    - probs : Normalized probabilities for each class,A numpy array of shape (n * number_of_classes)
    """
    logits = logits - np.max(logits, axis=0, keepdims=True)
    logits = np.exp(logits)
    return logits / np.sum(logits, axis=0, keepdims=True)



def cross_entropy_loss(probs,target):
    """
    Implement the cross Entropy loss function
    
    Inputs:
    - probs : A numpy-array of shape ( n * number_of_classes )
    - target : A numpy-array of shape ( n * number_of_classes )
    Returns:
    - loss : A scalar describing the mean cross-entropy loss over the batch
    """
    return np.mean(np.sum(-target*np.log(probs+epsilon), axis=-1))



def regularization_loss(weights,biases):
    """
    Inputs:
    - weights : weights of the network
    - biases : biases of the network
    
    Returns : the regularization loss
    """
    return 0.5*(np.sum(weights**2) + np.sum(biases**2))



def loss_fn(probs,target,weights,biases,_lambda):
    """
    function to calculate total loss
    Inputs:
    - probs : output of the neural network , a numpy array of shape (n, number_of_classes)
    - target : ground truth , a numpy array of shape (n,)
    - weights : weights of the network , an array containing weight matrices for all layers.(shape of the weight matrices vary according to the layer)
    - biases : biases of the network
    - _lambda : regularization constant
    Returns:
    - returns the total loss i.e - Mean cross-entropy loss + _lambda*regularization loss
    
    Note : This function is not being used anywhere.This will be used just for grading
    """
    return cross_entropy_loss(probs, target) + _lambda*regularization_loss(weights,biases)
    


def check_accuracy(prediction,target):
    """
    Find the accuracy of the prediction
    Inputs:
    - prediction : class probs for each datapoint, a numpy array of dimension (n, no_classes)
    - target : ground truth , a numpy array of dimension (n, no_classes)
    
    Returns :
    - accuracy : a scalar between 0 and 1 ,describing the accuracy , where 1 means prediction is same as ground truth
    
    """
    prediction = np.argmax(prediction, axis=-1)
    target = np.argmax(target, axis=-1)
    return np.mean(prediction==target)



class Neural_Net():
    def __init__(self,num_layers,num_units,input_dim,output_dim,initialization="randn"):
        '''
        Initialize the weights and biases of the network
        Inputs:
        - num_layers : Number of HIDDEN layers
        - num_units : Number of units in each hidden layer
        - input_dim : Number of features i.e your one batch is of shape (batch_size * input_dim)
        - output_dim : Number of units in output layer , i.e number of classes
        '''
        self.inputSize = input_dim  # Number of neurons in input layer
        self.outputSize = output_dim  # Number of neurons in output layer
        neurons = [input_dim] + [num_units] * num_layers + [output_dim]
        self.layers = len(neurons)
        self.weights = []  # weights for each layer
        self.biases = []  # biases in each layer

        if initialization == 'uniform':
            self.initializer = lambda h,w: np.random.uniform(-1,1,(h,w))
        elif initialization == 'randn':
            self.initializer = np.random.randn
        for i in range(len(neurons) - 1):
            self.weights.append(self.initializer(neurons[i + 1], neurons[i]))  # weight matrix between layer i and layer i+1
            self.biases.append(self.initializer(neurons[i + 1], 1))
        
        
    def forward(self, X):
        
        """
        Perform the forward step of backpropagation algorithm
        Inputs :
        - X : a numpy array of dimension (n , number_of_features)
        Returns :
        - out : the predictions for the data. For each training example, out
                 contains the probability distribution over all classes.
                 a numpy array of dimension (n , number_of-classes)
        - layer_dot_prod_z and layer_activations_a : archived z and a values for all layers
                 needed for backward pass of backpropagation

        """
        a = X.T # a is an array of shape (number_of_features, n)
        layer_activations_a = [a]  # store the input as the input layer activations
        layer_dot_prod_z = []
        for i, param in enumerate(zip(self.biases, self.weights)):
            b, w = param[0], param[1]
            z = np.dot(w, a) + b
            a = softmax(z) if i==self.layers-2 else ReLU(z)
            layer_dot_prod_z.append(z)
            layer_activations_a.append(a)
        out = a.T
        return out, layer_dot_prod_z, layer_activations_a


    
    def backward(self, X, y, zs, activations, _lambda):
        """
        perform the backward step of backpropagation algorithm and calculate the gradient of loss function with respect to weights and biases (dL/dW,dL/db)
        Inputs:
        - X : a single batch, a numpy array of dimension (n , number_of_features)
        - zs and activations: lists containing z and a values for all layers
        - probs : predictions for a single batch , a numpy array of dimension ( n, num_of_classes)
        - y : ground truth , a numpy array of dimension having dimension ( n, )
        - _lambda : regularization constant
        
        Returns:
        - grad_w - gradient of total loss with respect to weights,
        - grad_b - gradient of total loss with respect to biases,
        
        """
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        m = X.shape[0]

        # backward pass
        delta = activations[-1] - y.T # derivative of the crossentropy loss with respect of z of last layer

        # fill in the appropriate details for gradients of w and b
        grad_b[-1] = np.sum(delta, axis=1, keepdims=True) / m + _lambda*self.biases[-1]
        grad_w[-1] = np.dot(delta, activations[-2].T) / m + _lambda*self.weights[-1]

        for l in range(2, self.layers):  # Here l is in backward sense i.e. last l th layer
            z = zs[-l]
            # Compute delta, gradients of b and w
            delta = ReLUPrime(z) * np.dot(self.weights[-l + 1].T, delta)  # delta is dz
            grad_b[-l] = np.sum(delta, axis=1, keepdims=True) / m + _lambda*self.biases[-l]
            grad_w[-l] = np.dot(delta, activations[-l - 1].T) / m + _lambda*self.weights[-l]

        return (grad_b, grad_w)
    
    
    def train(self, optimizer, _lambda, batch_size, max_epochs,train_input, train_target,
              val_input, val_target, verbose=True):
        """
        Here you will run backpropagation for max_epochs number of epochs and evaluate 
        the neural network on validation data.For each batch of data in each epoch you 
        will do the forward step ,then backward step of backpropagation.And then you
        will update the gradients accordingly.

        Note : Most of the things here are already implemented.However, you are welcome to change it for part 2 of the assignment.
        """
        for epoch in range(max_epochs):
            idxs = np.arange(train_input.shape[0])
            np.random.shuffle(idxs) #shuffle the indices
            batch_idxs = np.array_split(idxs,np.ceil(train_input.shape[0]/batch_size)) #split into a number of batches

            for i in range(len(batch_idxs)):
                train_batch_input = train_input[batch_idxs[i],:] # input for a single batch
                train_batch_target = train_target[batch_idxs[i],:] # target for a single batch

                out, dot_prod_z, activations_a = self.forward(train_batch_input) # perform the forward step
                grads = self.backward(train_batch_input, train_batch_target, dot_prod_z,
                                      activations_a, _lambda) #perform the backward step and calculate the gradients

                self.weights,self.biases = optimizer.step(self.weights,self.biases,grads[1],grads[0]) # update the weights
            if epoch % 5 == 0 and verbose:
                train_probs,_,_ = self.forward(train_input)
                val_probs,_,_ = self.forward(val_input)
                train_loss = cross_entropy_loss(train_probs,train_target)
                val_loss = cross_entropy_loss(val_probs,val_target)
                train_acc = check_accuracy(self.predict(train_input),train_target)
                val_acc = check_accuracy(self.predict(val_input),val_target)
                print("train_loss = {:.3f}, val_loss = {:.3f}, train_acc={:.3f}, val_acc={:.3f}".format(train_loss,val_loss,train_acc,val_acc))

        if not verbose:
            train_probs, _, _ = self.forward(train_input)
            val_probs, _, _ = self.forward(val_input)
            train_loss = cross_entropy_loss(train_probs, train_target)
            val_loss = cross_entropy_loss(val_probs, val_target)
            # print("train_loss = {:.3f}, val_loss = {:.3f}".format(train_loss, val_loss))
            return train_loss, val_loss
                    
    def predict(self,X):
        """
        Predict the most probable classes for each datapoint in X
        Inputs : 
        - X : a numpy array of shape (n,number_of_features)
        Returns :
        - preds : Most probable class for each datapoint in X , a numpy array of shape (n,1)
        
        """
        preds, _, _ = self.forward(X)
        return preds


class Optimizer(object):

    def __init__(self, learning_rate):
        """
        Initialize the learning rate
        """
        self.lr = learning_rate

    def step(self, weights, biases, grad_w, grad_b):
        """
        update the gradients
        Inputs :
        - weights : weights of the network
        - biases : biases of the network
        - grad_w : gradients with respect to weights
        - grad_b : gradients with respect to biases
        Returns :
        Updated weights and biases
        """

        for i in range(len(weights)):
            weights[i] -= self.lr * grad_w[i]
            biases[i] -= self.lr * grad_b[i]

        return weights, biases

def get_csv_data(file_path):
    data = pd.read_csv(file_path)
    if 'letter' in data.columns: # indicator for whether ground-truth is available
        data["target"] = data["letter"].apply(lambda x: ord(x) - ord("A"))
        data = data.drop(columns=["letter"])
        data = data.to_numpy()
        return data[:, :-1], np.eye(26, dtype=int)[data[:,-1]] # targets are one-hot encoded
    else:
        return data.to_numpy()

def read_data():
    """
    Read the train,validation and test data
    """
    train_x, train_y = get_csv_data("data/train.csv")
    val_x, val_y = get_csv_data("data/val.csv")
    test_x = get_csv_data("data/test.csv")
    return train_x,train_y,val_x,val_y,test_x

def part_1b():
    data = pd.read_csv("part_1b.csv")
    train_x, train_y, val_x, val_y, test_x = read_data()
    for i in tqdm.tqdm(data.index):
        np.random.seed(42)
        max_epochs = 100
        batch_size = 128
        learning_rate = data.loc[i, "Learning Rate"]
        num_layers = data.loc[i, "No. of hidden layers"]
        num_units = data.loc[i, "Size of each hidden layer"]
        _lambda = data.loc[i, "λ(regulariser)"]
        net = Neural_Net(num_layers, num_units, train_x.shape[1], 26, initialization="uniform")
        optimizer = Optimizer(learning_rate=learning_rate)
        train_loss, val_loss = net.train(optimizer, _lambda, batch_size, max_epochs, train_x, train_y, val_x, val_y,
                                         verbose=False)
        data.loc[i, "Mean Cross Entropy loss(train)"] = np.round(train_loss, 3)
        data.loc[i, "Mean Cross Entropy loss(val)"] = np.round(val_loss, 3)
    data.to_csv("part_1b.csv", index=False)


if __name__ == '__main__':
    max_epochs = 100
    batch_size = 128
    learning_rate = 0.1
    num_layers = 1
    num_units = 64
    _lambda = 1e-5

    train_x,train_y,val_x,val_y,test_x = read_data()
    net = Neural_Net(num_layers,num_units,train_x.shape[1],26,initialization="uniform")
    optimizer = Optimizer(learning_rate=learning_rate)
    net.train(optimizer,_lambda,batch_size,max_epochs,train_x,train_y,val_x,val_y)

    test_preds = net.predict(test_x)

    # # Part 1-b
    # part_1b()
    

