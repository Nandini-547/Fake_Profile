app.py
from flask import Flask,render_template,url_for,request,json
from flask_bootstrap import Bootstrap
import numpy as np
#from collections import Counter
from pprint import pprint
import pickle
import sys
features=['statuses_count','followers_count','friends_count','favourites_count','listed_count','sex_
code','lang_code']
app = Flask( name )
Bootstrap(app)
@app.route('/')
def index():
                return render_template('index.html',features=enumerate(features))

@app.route('/predict', methods=['POST'])
def predict():
                 with open('pickle_model.pkl','rb') as f:
                 clf=pickle.load(f)
                # Receives the input query from form
                if request.method == 'POST':
                                namequery1 = request.form['namequery0']
                                namequery2 = request.form['namequery1']
                                namequery3 = request.form['namequery2']
                                namequery4 = request.form['namequery3']
                                namequery5 = request.form['namequery4']
                                namequery6 = request.form['namequery5']
                                namequery7 = request.form['namequery6']
                               data1=[]
                               data=[]
                               data.append(namequery1)
                              data.append(namequery2)
                              data.append(namequery3)
                              data.append(namequery4)
                             data.append(namequery5)
                            data.append(namequery6)
                            data.append(namequery7)
                            for i in data:
                                   data1.append(float(i))
                            vect = np.array(data1)
                            vect = vect.reshape(1,-1)
                            print(data, file=sys.stderr)
                            my_prediction = clf.predict(vect)
            return render_template('results.html',data=data,prediction =
my_prediction,max=0.01,labels=features, values=data1)
if name == ' main ':
          app.run(debug=True)