from flask import Flask, request, render_template, url_for, redirect
import pickle
import score

app = Flask(__name__)

model_loc="Applied-Machine-Learning/Assignments/models/lgr.pkl"
model=pickle.load(open(model_loc,"rb"))

threshold=0.7


@app.route('/') 
def home():
    return render_template('spam.html')


@app.route('/spam', methods=['POST'])
def spam():
    sent = request.form['sent']
    label,prop=score.score(sent,model,threshold)
    lbl="Spam" if label == 1 else "Not spam"
    ans = f"""The sentence "{sent}" is {lbl} with propensity {prop}."""
    return render_template('res.html', ans=ans)


if __name__ == '__main__': 
    app.run(debug=True)
