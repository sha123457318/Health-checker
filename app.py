from flask import Flask, redirect, url_for, request,render_template
import pickle
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

def ValuePredictor_dieabities(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 8)
    loaded_model = pickle.load(open("01.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result
def ValuePredictor_Kidney(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,18)
    loaded_model = pickle.load(open("Kidney.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result

def ValuePredictor_Heart(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,13)
    loaded_model = pickle.load(open("heart_1.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result
def ValuePredictor_Breast(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,22)
    X = pd.read_csv("1.csv")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print("************************************************************")
    print(X.shape)
    to_predict = scaler.transform(to_predict)
    loaded_model = pickle.load(open("cancer1.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result



@app.route('/')
def home():
   return render_template('index.html')
@app.route('/listening.html')
def listening():
   return render_template('listening.html')



@app.route('/dieabities',  methods =["GET"])
def dieabities():
    
       
    Pregnancies = int(request.args.get("Pregnancies"))
    # getting input with name = lname in HTML form 
    w = int(request.args.get("Glucose"))
    e = int(request.args.get("BloodPressure"))
    r = int(request.args.get("SkinThickness"))
    t = int(request.args.get("Insulin"))
    y = int(request.args.get("BMI"))
    u = int(request.args.get("PedigreeFunction"))
    i = int(request.args.get("Age")) 
        
    l1=[Pregnancies,w,e,r,t,y,u,i]
    ans = ValuePredictor_dieabities(l1)
    result = ""
    if ans[0]==1:
    	result = "Our Machine Learning model predict that , you have dieabities! "
    else:
    	result =" Chill, You safe ! "

    return render_template("listening.html",result=result)


@app.route('/Kidney',  methods =["GET"])
def Kidney():
    
       
    q = int(request.args.get("Age"))
    # getting input with name = lname in HTML form 
    w = int(request.args.get("BloodPressure"))
    e = int(request.args.get("Albumin"))
    r = int(request.args.get("Sugar"))
    t = int(request.args.get("RBC"))
    y = int(request.args.get("PC"))
    u = int(request.args.get("PCC"))
    i = int(request.args.get("Ba")) 
    o = int(request.args.get("BGR")) 
    p = int(request.args.get("Bu")) 
    a = int(request.args.get("sc")) 
    s = int(request.args.get("pu"))
    d = int(request.args.get("wc")) 
    f = int(request.args.get("Hypertension"))
    g = int(request.args.get("dm"))
    h = int(request.args.get("cad"))
    j = int(request.args.get("pe"))
    k = int(request.args.get("ane"))

    

        
    l1=[q,w,e,r,t,y,u,i,o,p,a,s,d,f,g,h,j,k]
    ans = ValuePredictor_Kidney(l1)

    result = ""
    if ans[0]==1:
    	result = "Our Machine Learning model predict that , you have chronic kidney disease.! "
    else:
    	result =" Chill, You safe ! "
    

    return render_template("listening.html",result=result)


@app.route('/Heart1',  methods =["GET"])
def Heart():
    
       
    q = int(request.args.get("age"))
    w = int(request.args.get("sex"))
    e = int(request.args.get("cp"))
    r = int(request.args.get("trestbps"))
    t = int(request.args.get("chol"))
    y = int(request.args.get("fsp"))
    u = int(request.args.get("restecg"))
    i = int(request.args.get("thalach")) 
    o = int(request.args.get("exang")) 
    p = int(request.args.get("oldpeak")) 
    a = int(request.args.get("slope")) 
    s = int(request.args.get("ca"))
    d = int(request.args.get("thal")) 

    

        
    l1=[q,w,e,r,t,y,u,i,o,p,a,s,d]
    ans = ValuePredictor_Heart(l1)

    result = ""
    if ans[0]==1:
    	result = "Our Machine Learning model predict that , you have Heart diseasei! "
    else:
    	result =" Chill, You safe ! "
    

    return render_template("listening.html",result=result)

@app.route('/Breast1',  methods =["GET"])
def Breast():
    
       
    q = int(request.args.get("texture_mean"))
    w = int(request.args.get("smoothness_mean"))
    e = int(request.args.get("compactness_mean"))
    r = int(request.args.get("concave_points_mean"))
    t = int(request.args.get("symmetry_mean"))
    y = int(request.args.get("FDM"))
    u = int(request.args.get("TSE"))
    i = int(request.args.get("ASE")) 
    o = int(request.args.get("SSE")) 
    p = int(request.args.get("CSE")) 
    a = int(request.args.get("CYSE")) 
    s = int(request.args.get("CPSE"))
    d = int(request.args.get("SSE")) 
    f = int(request.args.get("FDSE")) 
    g = int(request.args.get("TW")) 
    h = int(request.args.get("AW")) 
    j = int(request.args.get("SW")) 
    k = int(request.args.get("CW")) 
    l = int(request.args.get("CCW")) 
    z = int(request.args.get("CPW")) 
    x = int(request.args.get("SYW")) 
    c = int(request.args.get("FDW"))


    l1=[q,w,e,r,t,y,u,i,o,p,a,s,d,f,g,h,j,k,l,z,x,c]
    ans = ValuePredictor_Breast(l1)

    result = ""
    if ans[0]==1:
    	result = "Our Machine Learning model predict that , You Breast Cancer! "
    else:
    	result =" Chill, You safe ! "
    

    return render_template("listening.html",result= result)


if __name__ == '__main__':
   app.run(debug = True)




