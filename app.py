from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import random
import requests  # For making HTTP requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv

# Define your model class
class ModelClass(nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        self.fc1 = nn.Linear(in_features=10, out_features=50)  # Adjust input/output features
        self.fc2 = nn.Linear(in_features=50, out_features=20)
        self.fc3 = nn.Linear(in_features=20, out_features=6)  # Adjust for your output size

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation for the output layer 
        return x

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

def load_model(model_data):
    model = ModelClass()
    
    # Load the state dictionary into the model
    model.load_state_dict({k: torch.tensor(np.array(v)) for k, v in model_data.items()})
    
    model.eval()  # Set the model to evaluation mode
    return model

# Load items from CSV file
def load_items():
    df = pd.read_csv('items.csv')
    items = df.to_dict(orient='records')  # Convert DataFrame to a list of dictionaries
    return items

# Load user ratings from dummy_data.csv
def load_ratings():
    df = pd.read_csv('dummy_data.csv')
    ratings = df.to_dict(orient='records')  # Convert DataFrame to a list of dictionaries
    return ratings

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    user_list = list(pd.read_csv("dummy_data.csv")["user_id"].unique())
    user_id = session.get('user_id')

    # if not user_id:
    #    return redirect(url_for('index'))  # Redirect if user is not logged in

    # uncomment them when the http request is avaliable
    lambda_url = 'https://hco230csm8.execute-api.us-east-1.amazonaws.com/default/ipfs-blockchain-federated-pipeline' 
    response = requests.post(lambda_url)
    response_state = response.status_code

    response_state = 200  # remove this when http request is avaliable
    if response_state == 200:

        model_data = response.json()  # Get the model data from the Lambda response
        
        # Load the model with the retrieved data
        model = load_model(model_data)

        # Prepare input data for the model (modify according to your specific needs)
        input_data = torch.tensor([user_id], dtype=torch.float32)  # Adjust based on actual input shape

        # Run inference
        with torch.no_grad():
            recommendations = model(input_data) # Inference

        # Process recommendations (convert to a list)
        recommendations_list = recommendations.numpy().tolist()  # Adjust as necessary
        """
        # Assume the model can return array of product_ids sorted from most relevant to least relevant
        # Create the array
        arr = np.array([1, 2, 3, 4, 5, 6]) # for illustration, we didnt invoke the model in the front end

        # Shuffle the array in place
        np.random.shuffle(arr)

        # Get the first 3 elements
        recommendations_list = arr[:3]
        """

        # Fetch item details based on recommendations
        selected_items = fetch_items_by_recommendations(recommendations_list)
        
        return render_template('recommendation.html', selected_items=selected_items)
    else:
        flash('Failed to get model from Lambda, please try again later.', 'error')
        return redirect(url_for('personal'))  # Redirect back to the personal page

@app.route('/login', methods=['POST'])
def login():
    user_list = list(pd.read_csv("dummy_data.csv")["user_id"].unique())
    user_id = request.form.get('user_id')
    if user_id:
        try:
            user_id = int(user_id)
            if user_id < 0:
                flash('Please enter a valid USER ID (non-negative integer only).', 'error')
                return redirect(url_for('index'))
            elif user_id not in user_list:
                flash(f'Please enter a valid USER ID (User ID {user_id} not exist!).', 'error')
                return redirect(url_for('index'))
            session['user_id'] = user_id
            return redirect(url_for('personal'))  # Make sure this is correct
        except ValueError:
            flash('Please enter a valid user ID (non-negative integer only).', 'error')
            return redirect(url_for('index'))
    return redirect(url_for('index'))


@app.route('/contribute')
def contribute():
    # Load items from items.csv
    items = pd.read_csv('items.csv').to_dict(orient='records')  # Load items if needed
    # Load client data from clientdata.csv
    client_data = pd.read_csv('clientdata.csv').to_dict(orient='records')
    return render_template('contribute.html', items=items, client_data=client_data)

@app.route('/home')
def home():
    session.pop('user_id', None)  # Clear the user_id from the session
    return redirect(url_for('index'))

@app.route('/personal')
def personal():
    all_items = load_items()  # Load all items from the dataset
    featured_items = random.sample(all_items, min(3, len(all_items)))  # Select 3 random items
    return render_template('personal.html', featured_items=featured_items)


def fetch_items_by_recommendations(recommendations):
    items = load_items()  # Load all items
    selected_items = [item for item in items if item['item_id'] in recommendations]
    return selected_items


@app.route('/submit_contribution', methods=['POST'])
def submit_contribution():
    user_ids = request.form.getlist('user_id[]')  # Get the user IDs from the form
    event_times = request.form.getlist('event_time[]')  # Get all event times
    event_types = request.form.getlist('event_type[]')  # Get all event types
    product_ids = request.form.getlist('product_id[]')  # Get all product IDs
    category_ids = request.form.getlist('category_id[]')  # Get all category IDs
    category_codes = request.form.getlist('category_code[]')  # Get all category codes
    brands = request.form.getlist('brand[]')  # Get all brands
    prices = request.form.getlist('price[]')  # Get all prices
    user_sessions = request.form.getlist('user_session[]')  # Get all user session IDs

    # Prepare data to be written to CSV
    contributions = []

    for user_id, event_time, event_type, product_id, category_id, category_code, brand, price, user_session in zip(
            user_ids, event_times, event_types, product_ids, category_ids, category_codes, brands, prices, user_sessions):
        contributions.append({
            'user_id': user_id,  # Use the user_id from the form
            'event_time': event_time,
            'event_type': event_type,
            'product_id': product_id,
            'category_id': category_id,
            'category_code': category_code,
            'brand': brand,
            'price': price,
            'user_session': user_session
        })

    # Append contributions to dummy_data.csv
    with open('dummy_data.csv', 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=contributions[0].keys())
        if f.tell() == 0:  # Check if file is empty and write header
            writer.writeheader()
        writer.writerows(contributions)

    url = "https://hco230csm8.execute-api.us-east-1.amazonaws.com/default/ipfs-blockchain-federated-pipeline"
    payload = {
        "updatedClientIndex": "0",
        "blockchainEnabled": "1",
        "modelEncoded" : "UEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAQABIAYXJjaGl2ZS9kYXRhLnBrbEZCDgBaWlpaWlpaWlpaWlpaWoACY2NvbGxlY3Rpb25zCk9yZGVyZWREaWN0CnEAKVJxAShYCQAAAGZjLndlaWdodHECY3RvcmNoLl91dGlscwpfcmVidWlsZF90ZW5zb3JfdjIKcQMoKFgHAAAAc3RvcmFnZXEEY3RvcmNoCkZsb2F0U3RvcmFnZQpxBVgBAAAAMHEGWAMAAABjcHVxB0sKdHEIUUsASwFLCoZxCUsKSwGGcQqJaAApUnELdHEMUnENWAcAAABmYy5iaWFzcQ5oAygoaARoBVgBAAAAMXEPaAdLAXRxEFFLAEsBhXERSwGFcRKJaAApUnETdHEUUnEVdX1xFlgJAAAAX21ldGFkYXRhcRdoAClScRgoWAAAAABxGX1xGlgHAAAAdmVyc2lvbnEbSwFzWAIAAABmY3EcfXEdaBtLAXN1c2IuUEsHCLgNEJ06AQAAOgEAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAADgAKAGFyY2hpdmUvZGF0YS8wRkIGAFpaWlpaWvavUL26qJ4921mpvSdHkT4YwWi+b4RqPdPAhT6Pr4O+J9sJvgb2MT5QSwcIRey7MigAAAAoAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAOABwAYXJjaGl2ZS9kYXRhLzFGQhgAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaYj9ZPVBLBwg16E2sBAAAAAQAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAA8APwBhcmNoaXZlL3ZlcnNpb25GQjsAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlozClBLBwjRnmdVAgAAAAIAAABQSwECAAAAAAgIAAAAAAAAuA0QnToBAAA6AQAAEAAAAAAAAAAAAAAAAAAAAAAAYXJjaGl2ZS9kYXRhLnBrbFBLAQIAAAAACAgAAAAAAABF7LsyKAAAACgAAAAOAAAAAAAAAAAAAAAAAIoBAABhcmNoaXZlL2RhdGEvMFBLAQIAAAAACAgAAAAAAAA16E2sBAAAAAQAAAAOAAAAAAAAAAAAAAAAAPgBAABhcmNoaXZlL2RhdGEvMVBLAQIAAAAACAgAAAAAAADRnmdVAgAAAAIAAAAPAAAAAAAAAAAAAAAAAFQCAABhcmNoaXZlL3ZlcnNpb25QSwYGLAAAAAAAAAAeAy0AAAAAAAAAAAAEAAAAAAAAAAQAAAAAAAAA8wAAAAAAAADSAgAAAAAAAFBLBgcAAAAAxQMAAAAAAAABAAAAUEsFBgAAAAAEAAQA8wAAANICAAAAAA=="
    }
    response = requests.post(url, json=payload)
    
    flash('Contributions submitted successfully!', 'success')
    return redirect(url_for('personal'))  # Redirect back to the personal dashboard
if __name__ == '__main__':
    app.run(debug=True)
