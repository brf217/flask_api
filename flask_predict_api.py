import pickle
from flask import Flask, request
import numpy as np
import pandas as pd
from flasgger import Swagger

'''http://localhost:5000/apidocs/    - how to see flask api ui'''

#open pickle file and make sure to tell it is a binary 'rb' read
with open ('/Users/feebr01/Documents/p_docker/rf.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    
app = Flask(__name__)
swagger = Swagger(app)  #make ui look nice

@app.route('/predict')
def predict_iris():
    '''Example enpoint returning a prediction of iris
    ---
    parameters:
        - name: s_length
          in: query
          type: number
          required: true
    
        - name: s_width
          in: query
          type: number
          required: true    
     
        - name: p_length
          in: query
          type: number
          required: true    

        - name: p_width
          in: query
          type: number
          required: true
    responses:
        '200':
            description: "text"
    '''   
    s_length = request.args.get('s_length')
    s_width = request.args.get('s_width')
    p_length = request.args.get('p_length')
    p_width = request.args.get('p_width')
    
    
    prediction = model.predict(np.array([[s_length,s_width,
                                          p_length,p_width]]))
    
    return str(prediction)

@app.route('/predict_file', methods = ['POST'])
def predict_iris_file():
    '''Example file enpoint returning a prediction of iris
    ---
    parameters:
        - name: input_file
          in: formData
          type: file
          required: true
    responses:
        '200':
            description: "text"
    '''
    input_data = pd.read_csv(request.files.get('input_file'), header=None)
    prediction = model.predict(input_data)
    return str(list(prediction))

if __name__ == '__main__':
    app.run()
    

    
    
    
    
