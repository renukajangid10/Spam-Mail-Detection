from flask import Flask,render_template,url_for,request
import pickle
import spam_detect_model as sdm

app = Flask(__name__)

@app.route('/')
def home():
        return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
        g = open('spam_model.pkl','rb')
        spam_model2 = pickle.load(g)
        g.close()
        if request.method == 'POST':
                message = request.form['message']
                data = message 
                my_prediction = spam_model2.classify(sdm.Features_Extraction(sdm.process(data),''))
        
        return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
        app.run(debug=True)
