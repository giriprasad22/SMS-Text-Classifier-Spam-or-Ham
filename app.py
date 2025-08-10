from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load model
model = joblib.load('spam_model.joblib')

def predict_message(text):
    prob = model.predict_proba([text])[0]
    prediction = model.predict([text])[0]
    return {
        'label': prediction,
        'spam_probability': float(prob[1]),
        'ham_probability': float(prob[0])
    }

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        text = request.form['message']
        result = predict_message(text)
    return render_template('index.html', result=result, max=max)  # Add max here

if __name__ == '__main__':
    app.run(debug=True)