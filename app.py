# https://youtu.be/l3QVYnMD128
"""
Application that predicts heart disease percentage in the population of a town
based on the number of bikers and smokers. 

Trained on the data set of percentage of people biking 
to work each day, the percentage of people smoking, and the percentage of 
people with heart disease in an imaginary sample of 500 towns.

"""


import numpy as np
from flask import Flask, request, render_template
import pickle, os

#Create an app object using the Flask class. 
app = Flask(__name__)

#Load the trained model. (Pickle file)
#model = pickle.load(open('models/model.pkl', 'rb'))

#Define the route to be home. 
#The decorator below links the relative route of the URL to the function it is decorating.
#Here, home function is with '/', our root directory. 
#Running the app sends us to index.html.
#Note that render_template means it looks for the file in the templates folder. 

#use the route() decorator to tell Flask what URL should trigger our function.
@app.route('/')
def home():
    return render_template('index.html')


def preprocessing(file):

    import pandas as pd
    df = pd.read_csv(file)
    
    print(df.columns)
    
    print(df.head())
    
    print(df['Star'].isna().sum(), df['Text'].isna().sum())
    
    df = df[df['Text'].notnull()]
    
    print(df['Star'].isna().sum(), df['Text'].isna().sum())
    
    print(df['Star'].value_counts())
    
    #!pip install happytransformer
    
    from happytransformer import HappyTextClassification
    
    classifier = HappyTextClassification(model_type="DISTILBERT", model_name="distilbert-base-uncased-finetuned-sst-2-english", num_labels=2)
    
    print(df['Text'].values)
    
    results = []
    for i in df['Text'].values:
        results.append(classifier.classify_text(i))
    
    results = [1 if res.label=='POSITIVE' else 0 for res in results]
    
    print(results[:10])
    
    df['sentiment'] = results
    
    """**Considering reviews less than 2.5 as negative reviews**"""
    
    final_result = df[(df['sentiment']==1) & (df['Star']<2.5)][['Text', 'Star']]
    return final_result

#You can use the methods argument of the route() decorator to handle different HTTP methods.
#GET: A GET message is send, and the server returns data
#POST: Used to send HTML form data to the server.
#Add Post method to the decorator to allow for form submission. 
#Redirect to /predict page with the output
@app.route('/predict',methods=['POST'])
def predict():
    file = request.files["csv_file"]
    file.save(os.path.join("uploads", file.filename))
    result_df = preprocessing(os.path.join("uploads", file.filename))
    print(result_df)
#    int_features = [float(x) for x in request.form.values()] #Convert string inputs to float.
#    features = [np.array(int_features)]  #Convert to the form [[a, b]] for input to the model
#    prediction = model.predict(features)  # features Must be in the form [[a, b]]
#
#    output = round(prediction[0], 2)
    df_html = result_df.to_html()
    return f'{df_html}'
    return render_template('index.html', prediction_text=f'Faulty reviews are {result_df}')


#When the Python interpreter reads a source file, it first defines a few special variables. 
#For now, we care about the __name__ variable.
#If we execute our code in the main program, like in our case here, it assigns
# __main__ as the name (__name__). 
#So if we want to run our code right here, we can check if __name__ == __main__
#if so, execute it here. 
#If we import this file (module) to another file then __name__ == app (which is the name of this python file).

#if __name__ == "__main__":
#    app.run()
