import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 


class Digit_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128) # 28x28 pixels means 784 float values for the first layer connected to 64 neurons
        self.fc2 = nn.Linear(128, 64)  # connect with another 128 neurons
        self.fc3 = nn.Linear(64, 10)   # connect to output layer
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_fn   = nn.CrossEntropyLoss()
        
    def forward(self, x):
        x = x.flatten(start_dim=1) # turn the 64 batches of 1 chanel 28x28 matricies from (64,1,28,28) -> (64,784)
                                   # start_dim means "flatten all dimensions after dim=1"
        x = F.relu(self.fc1(x))    # feed input as a vector, map it to 128 neurons: (64, 784) -> (64, 128)
        x = F.relu(self.fc2(x))    # map the second layer to the third layer: (64,128) -> (64, 64)
        x = self.fc3(x) #map the last hidden layer to the output layer using raw logits because loss function handles activation
        
        return x                   #where x is a probability distribution for 10 vlaues.
    
    def fit(self, loader, epochs=20):

        for i in range(epochs):
            self.train() # put model in training mode
            for x, y in loader: # x is the batch or vector of input values for each image | y is correct prediction vec
                self.optimizer.zero_grad()      # clear gradients beofre recalculation
                loss = self.loss_fn(self(x), y) # self(x) constructs the graph based on x and runs forward, we 
                                                # calculate the loss of constructed graph's preds vs correct pred
                loss.backward()                 # this backpropogates the gradients for all the wieghts
                self.optimizer.step()           # this is what changes the weights and marks the end of the loop steps
    
    @torch.no_grad() # wraps the function in a method that says "don't compute gradients to save time"
    def evaluate(self, loader):
        in_training = self.training
        self.eval()
        correct = 0
        for x, y in loader:
            preds = torch.argmax(self(x), dim=1)   # (64,) predicted digits
            correct += (preds == y).sum().item()   # count exact matches
        
        self.train(in_training)                  #if evaluating in the middle of training, set mode back to training
        return correct/len(loader.dataset)