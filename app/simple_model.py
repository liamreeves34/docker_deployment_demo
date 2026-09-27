import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 


class Digit_Classifier(nn.Module):
    def __init__(self):
        super(Digit_Classifier, self).__init__()
        
        self.fc1 = nn.Linear(784, 64) #28x28 pixels means 784 float values for the first layer connected to 64 neurons
        self.fc2 = nn.Linear(64, 64)  #connect with another 64 neurons
        self.fc3 = nn.Linear(64, 10)  #connect to output layer
        
    def forward(self, x):
        x = F.relu(self.fc1(x))    #feed input as a vector, connect it to the graph, assigns input/first layer of graph
        x = F.relu(self.fc2(x))    #connect input/layer1 of graph to second layer
        x = F.softmax(self.fc3(x)) #connect previous layers to output layers, but use softmax for probabilities
        
        return x                   #where x is the fully connected graph with activation functions applied.
    
    def fit(self, loader, epochs=20, learnrate = 0.001):
        self.optimizer = optim.Adam(self.parameters(), lr=learnrate)
        self. loss_fn   = nn.CrossEntropyLoss()
        
        for i in range(epochs):
            self.train() # put model in training mode
            for x, y in loader: # x is the batch or vector of input values for each image | y is correct prediction vec
                self.optimizer.zero_grad()      # clear gradients beofre recalculation
                loss = self.loss_fn(self(x), y) # self(x) constructs the graph based on x and runs forward, we 
                                                # calculate the loss of constructed graph's preds vs correct pred
                loss.backwards()                # this backpropogates the gradients for all the wieghts
                self.optimizer.step()           # this is what changes the weights and marks the end of the loop steps
    
    @torch.no_grad() #wraps the function in a method that says "don't compute gradients to save time"
    def evaluate(self, loader):
        self.eval()
        x, y  = loader
        preds = self(x)
        loss  = self.loss_fn(self(x), y)