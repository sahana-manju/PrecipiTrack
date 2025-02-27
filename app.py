from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
#import jsonify
#import requests
#import pickle
import numpy as np
import pandas as pd
#import sklearn
#from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from integration import Preprocess_LSTM,Preprocess_RIDGE,Preprocess_XGB
import sys
import joblib
from joblib import load
sys.modules['sklearn.externals.joblib'] = joblib
from xgboost import XGBRegressor
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers.legacy import Nadam  # Use legacy optimizer for M1/M2 Mac
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, LSTM, GlobalAveragePooling1D, AveragePooling1D,
    TimeDistributed, Flatten, Bidirectional, Dropout, Masking, Layer,
    BatchNormalization
)
print(tf.__version__)

# Clear existing custom objects
tf.keras.saving.get_custom_objects().clear()

# Register Custom Layer
@tf.keras.saving.register_keras_serializable(package="MyLayers")
class NonMasking(Layer):   
    def __init__(self, **kwargs):   
        self.supports_masking = True  
        super(NonMasking, self).__init__(**kwargs)   

    def call(self, x, mask=None):   
        return x   

    def compute_output_shape(self, input_shape):   
        return input_shape 

# Load model without compiling
rmodel = load_model("lstm_custom_model.keras", custom_objects={"NonMasking": NonMasking}, compile=False)

# Compile it manually
rmodel.compile(
    optimizer=Nadam(learning_rate=0.001),  # Use legacy Nadam
    loss="mse",
    metrics=["mae"]
)

class RidgeRegScratch():
  
  def __init__(self, alpha=1.0, solver='closed'):
      self.alpha = alpha
      self.solver = solver

  def fit(self, X, y):
      X_with_intercept = np.c_[np.ones((X.shape[0], 1)), X]
      self.X_intercept = X_with_intercept
      if self.solver == 'closed':
          dimension = X_with_intercept.shape[1]
          A = np.identity(dimension)
          A[0, 0] = 0
          A_biased = self.alpha * A
          thetas = np.linalg.inv(X_with_intercept.T.dot(
              X_with_intercept) + A_biased).dot(X_with_intercept.T).dot(y)
      self.thetas = thetas
      return self

  def predict(self, X):
      thetas = self.thetas
      X_predictor = np.c_[np.ones((X.shape[0], 1)), X]
      self.predictions = X_predictor.dot(thetas)
      return self.predictions

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']="sqlite:///test.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=False
db=SQLAlchemy(app)
class update_rain(db.Model):
    mid=db.Column(db.Integer,primary_key=True)
    date=db.Column(db.Integer,nullable=False)
    month=db.Column(db.String(200),nullable=False)
    year=db.Column(db.Integer,nullable=False)
    rainfall=db.Column(db.Float,nullable=False)
    
    def __repr__(self)->str:
        return f"{self.mid}-{self.title}"
#model = load_model('rnn2.h5')
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    
    if request.method == 'POST':
        error_msg1=''
        error_msg2=''
        error_msg3=''
        
        if request.form['mid'] == '':
            error_msg1='Please enter meteorologist id'
        else:
            mid=request.form['mid']
        if request.form['date']=='':
            error_msg2='Please enter date'
        else:
            date1=request.form['date']
            print(date1)
            date=date1[9:]
            month=date1[5:7]
            year=date1[0:4]
               

        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            error_msg3='No file provided'
        else:
            uploaded_file.save(uploaded_file.filename)
        if error_msg1!=''or error_msg2 !='' or error_msg3!='':
            print("i reached here 1")
            return render_template('index.html',error_msg1=error_msg1,
                                   error_msg2=error_msg2,
                                   error_msg3=error_msg3
                                   )
        else:
            print("i reached here 2")
            test_raw1=pd.read_csv(uploaded_file.filename, index_col=0)
            test_raw1.fillna(0)
            l1=list(test_raw1.columns)
            print(l1)
            print(len(l1))
            l2=[]
            float_array=np.array([1,2.0])
            int_array=np.array([1,9223372036854775807])
            float_t=type(float_array[0])
            int_t=type(int_array[0])
            if len(test_raw1.columns) != 23:
                return render_template('index.html',error_msg4='No of columns should be 23')
            # for i in l1:
            #     if i == 'Id' or i =='minutes_past' or i=='radardist_km':
            #         if type(test_raw1[i].iloc[0]) != int_t:
            #             print(type(test_raw1[i].iloc[0]))
            #             s1=i+' is not in right format. Required format is '+str(int_t)
            #             l2.append(s1)
            #     else:
            #         if type(test_raw1[i].iloc[0]) != float_t:
            #             s2=i+' is not in right format. Required format is '+str(float_t)
            #             l2.append(s2)
            if len(l2)!=0:
                return render_template('index.html',error_msg5=l2)
            else:
                
                #LSTM
                pre=Preprocess_LSTM(test_raw1)
                test_new=pre.handle_missing_values(test_raw1)
                sc1=load('rnn2.bin')
                test_new=pre.scale_transform(test_new,sc1)
                X_test=pre.create_dataset(test_new)
                #model=load_model('masked_lstm_v3.h5')
                # Load the model (ensure your custom layer is included)
                #model = load_model('masked_lstm_v3.h5', custom_objects={'NonMasking': NonMasking})
                output1=rmodel.predict(X_test, batch_size=64,verbose=1)
                del test_new,X_test
                
                #RIDGE
                pre_ridge=Preprocess_RIDGE(test_raw1)
                test_new1=pre_ridge.create_dataset(test_raw1)
                test_new1=pre_ridge.handle_missing_values(test_new1)
                sc2=load('ridge.bin')
                test_new1=pre_ridge.scale_transform(test_new1,sc2)
                ridge_model= joblib.load('ridge.sav')
                output2=ridge_model.predict(test_new1)
                del test_new1
                
                #XGB
                pre_xgb=Preprocess_XGB(test_raw1)
                test_raw2=pre_xgb.handle_missing_values(test_raw1)
                test_new2=pre_xgb.create_dataset(test_raw2)
                sc2=load('xgb_std2.bin')
                test_new2=pre_xgb.scale_transform(test_new2,sc2)
                pc2=load('xgb_pca2.bin')
                test_new2=pre_xgb.pca_transform(test_new2,pc2)
                model2 = XGBRegressor()
                model2.load_model("model_xgb.txt")
                output3=model2.predict(test_new2)
                
                
                #INTEGRATION
                new_out=np.zeros(len(output1))
                for i in range(len(output1)):
                    if output1[i]<5:
                        new_out[i]=(output1[i]+output2[i])/2
                    else:
                        new_out[i]=output1[i]+output2[i]+output3[i]
                output=new_out.sum()
                output=np.round(output,2)
                #rain=update_rain(mid=mid,date=date,month=month,year=year,rainfall=output)
                #db.session.add(rain)
                #db.session.commit()
                return render_template('index.html',prediction_text="Total Rainfall in mm {}".format(output))
                

        
    else:
        return render_template('index.html',prediction_text="no file sorry")

if __name__=="__main__":
    app.run(debug=True)