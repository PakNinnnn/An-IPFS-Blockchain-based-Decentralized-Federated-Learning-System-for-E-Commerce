import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader 

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
        

def infer(model_path, user, num_items):
    max_user_id = 7006 + 1
    max_product_id = 9423 + 1
    model = MatrixFactorization(max_user_id, max_product_id, n_factors=100)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    prediction = model(torch.Tensor(user), torch.arange(0, max_product_id))
    return torch.topk(prediction, num_items).to_numpy()

def lambda_handler(event, context):
    # TODO implement
    json_input = json.loads(event['body'])

    top_items = infer(json_input['model_path'], json_input['user'], json_input['num_items'])
    top_items = np.array(top_items)

    return {
        'statusCode': 200,
        'body': top_items
    }
