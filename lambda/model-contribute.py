import json
import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import Dataset, DataLoader 
import random

# Source: modified from https://medium.com/@rinabuoy13/explicit-recommender-system-matrix-factorization-in-pytorch-f3779bb55d74

class RatingDataset(Dataset):
    def __init__(self, train, label):
        self.feature_ = train
        self.label_ = label  
    def __len__(self):
        return len(self.feature_)  
    def __getitem__(self, idx):
        return torch.tensor(self.feature_[idx]),torch.tensor(self.label_[idx])
    
class MatrixFactorization(torch.nn.Module):

    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors)
        self.item_factors = torch.nn.Embedding(n_items, n_factors)
        self.user_biases = torch.nn.Embedding(n_users, 1)
        self.item_biases = torch.nn.Embedding(n_items,1)
        torch.nn.init.xavier_uniform_(self.user_factors.weight)
        torch.nn.init.xavier_uniform_(self.item_factors.weight)
        self.user_biases.weight.data.fill_(0.)
        self.item_biases.weight.data.fill_(0.)
        
    def forward(self, user, item):
        pred = self.user_biases(user) + self.item_biases(item)
        pred += (self.user_factors(user) * self.item_factors(item)).sum(1, keepdim=True)
        return pred.squeeze()
        

def train_model(inp_path, sav_path):
    
    # Load csv
    data = pd.read_csv(inp_path)

    # First, preprocess the data using dataframe

    max_user_id = 7006 + 1
    max_product_id = 9423 + 1

    def event_test(type):
        if type == 'view':
            return 0 + random.random()
        if type == 'cart':
            return 1 + random.random()
        if type == 'purchase':
            return 3 + random.random()
        return 2 + random.random()

    # simplified ratings
    interaction = data['event_type'].apply(event_test)
    data['event'] = data['event_type'].apply(event_test)
    data['interaction'] = interaction
    used_data = data[['user_id','product_id','interaction']]

    # Next, PyTorch
    torch_data = torch.zeros(max_user_id, max_product_id)
    for item in used_data.index:
        torch_data[used_data.loc[item, 'user_id'], used_data.loc[item, 'product_id']] = used_data.loc[item, 'interaction']

    labels = torch.Tensor(used_data.loc[:, 'interaction'].to_numpy())

    nfactor = 100
    model = MatrixFactorization(max_user_id, max_product_id, n_factors=nfactor)

    X_train, y_train = torch_data[0:9000], labels[0:9000]
    X_test, y_test = torch_data[9001:], labels[9001:]

    bs = 1000
    train_dataloader = DataLoader(RatingDataset(X_train, y_train), batch_size=bs, shuffle=True)
    test_dataloader = DataLoader(RatingDataset(X_test, y_test), batch_size=bs)

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    loss_func = torch.nn.MSELoss()
    model.to(dev)
    epoches = 10
    for epoch in range(0,epoches):
        pbar = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))  # progress bar
        count = 0
        cum_loss = 0.
        train_loss = 0.
        test_loss = 0.
        for i,(train_batch, label_batch) in pbar:
            count = 1 + i
            # Predict and calculate loss for user factor and bias
            optimizer = torch.optim.SGD([model.user_biases.weight,model.user_factors.weight], lr=0.01, weight_decay=1e-5) # learning rate
            XUser = train_batch[:,0].to(dev, dtype=torch.long)
            XProduct = train_batch[:,1].to(dev, dtype=torch.long)
            prediction = model(XUser, XProduct)
            loss = loss_func(prediction, label_batch.to(dev))    
            # Backpropagate
            loss.backward()
            
            # Update the parameters
            optimizer.step()
            optimizer.zero_grad()
            
            #predict and calculate loss for item factor and bias       
            optimizer = torch.optim.SGD([model.item_biases.weight,model.item_factors.weight], lr=0.01, weight_decay=1e-5) # learning rate
            XUser = train_batch[:,0].to(dev, dtype=torch.long)
            XProduct = train_batch[:,1].to(dev, dtype=torch.long)
            prediction = model(XUser, XProduct)
            loss = loss_func(prediction, label_batch.to(dev))
            # Backpropagate
            loss.backward()
            
            # Update the parameters
            optimizer.step()
            optimizer.zero_grad()
            cum_loss += loss.item()
            pbar.set_description('training loss at {} batch {}: {}'.format(epoch,i,loss.item()))

            train_loss = cum_loss/count
            
            pbar = tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader))  

        for i,(test_batch, label_batch) in pbar:
            count = 1 + i
            with torch.no_grad():
                user_test = test_batch[:,0].to(dev, dtype=torch.long)
                product_test = test_batch[:,1].to(dev, dtype=torch.long)
                prediction = model(user_test, product_test)
                loss = loss_func(prediction, label_batch.to(dev))
                cum_loss += loss.item()
                pbar.set_description('test loss at {} batch {}: {}'.format(epoch,i,loss.item()))
            test_loss = cum_loss/count

    # Save to path
    torch.save(model.state_dict(), f'{sav_path}/user_trained.pth')

def lambda_handler(event, context):
    # TODO implement
    json_input = json.loads(event['body'])

    train_model(json_input['data_path'], json_input['save_path'])

    return {
        'statusCode': 200,
        'body': json.dumps('Model Trained')
    }
