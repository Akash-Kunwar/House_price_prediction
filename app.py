import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import dump, load

app = Flask(__name__)
model = load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Chances of getting house is around {} %'.format(output))



@app.route('/api',methods=['GET'])
def predict_api():
    crim = request.args.get('crim')
    zin= request.args.get('zin')
    indus= request.args.get('indus')
    chas= request.args.get('chas')
    nox= request.args.get('nox')
    rm= request.args.get('rm')
    age= request.args.get('age')
    dis= request.args.get('dis')
    rad= request.args.get('rad')
    tax= request.args.get('tax')
    ptratio= request.args.get('ptratio')
    b= request.args.get('b')
    lstat= request.args.get('lstat')
    
    int_features = []
    int_features.append(int(crim))
    int_features.append(int(zin))
    int_features.append(int(indus))
    int_features.append(int(chas))
    int_features.append(int(nox))
    int_features.append(int(rm))
    int_features.append(int(age))
    int_features.append(int(dis))
    int_features.append(int(rad))
    int_features.append(int(tax))
    int_features.append(int(ptratio))
    int_features.append(int(b))
    int_features.append(int(lstat))

    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    print('prediction',prediction)
    return {
        "price": prediction[0],
    }

if __name__ == "__main__":
    app.run(debug=True)