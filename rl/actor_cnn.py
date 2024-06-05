import torch
import torch.nn as nn
import torch.autograd as autograd 
import torch.nn.functional as F
from torch.distributions import Categorical

#Predicts action
class ActorCnn(nn.Module):
    def __init__(self, input_shape, pos_shape, lie_shape):
        super(ActorCnn, self).__init__()
        self.input_shape = input_shape
        self.pos_shape = pos_shape
        self.lie_shape = lie_shape
        
        # NN for interpreting image
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # NN for choice of Theta (1-360)
        self.theta_fc = nn.Sequential(
            nn.Linear(self.feature_size() + pos_shape[1] + lie_shape, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        # NN for choice of club (1-14)
        self.club_fc = nn.Sequential(
            nn.Linear(self.feature_size() + pos_shape[1] + lie_shape, 512),
            nn.ReLU(),
            nn.Linear(512, 14),
            nn.Softmax(dim=1)
        )
        
    def forward(self, state):
        img, pos, lie = state # get image, position, and lie
        x = self.features(img) # get feature of img
        x = x.view(x.size(0), -1)
        x = torch.cat((x, pos.squeeze(), lie), dim=1) # prepare x to pass in to NNs
        
        theta = self.theta_fc(x) * 360 # multiply the output of theta_fc (which is between 0 and 1) by 360 to get an angle
        club_probs = self.club_fc(x) # get prediction for club
        
        club_dist = Categorical(club_probs)
        
        return theta, club_dist
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)


#Predicts value
class CriticCnn(nn.Module):
    def __init__(self, input_shape, pos_shape, lie_shape):
        super(CriticCnn, self).__init__()
        self.input_shape = input_shape
        self.pos_shape = pos_shape
        self.lie_shape = lie_shape
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size() + pos_shape[1] + lie_shape, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
    def forward(self, state):
        img, pos, lie = state
        x = self.features(img)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, pos.squeeze(), lie), dim=1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)