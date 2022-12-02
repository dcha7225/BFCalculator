from flask import Flask, render_template, redirect, url_for, request
import pickle
import numpy as np

app = Flask(__name__)

@app.route("/", methods = ["POST", "GET"])
def home():
    if request.method == "POST":
        age = request.form["age"]
        height = request.form["height"]
        weight = request.form["weight"]
        neck = request.form["neck"]
        chest = request.form["chest"]
        ab = request.form["ab"]
        hip = request.form["hip"]
        thigh = request.form["thigh"]
        knee = request.form["knee"]
        ankle = request.form["ankle"]
        bicep = request.form["bicep"]
        forearm = request.form["forearm"]
        wrist = request.form["wrist"]

        x = (age, height, weight, neck, chest, ab, hip, thigh, knee, ankle, bicep, forearm, wrist)
        for entry in x:
            if not entry.isdigit():
                return render_template("index.html")

        tree = pickle.load(open('finalized_model7.sav', 'rb'))
        test = np.array([age, height, weight, neck, chest, ab, hip, thigh, knee, ankle, bicep, forearm, wrist])
        
        result = str(tree.predict(test.reshape(1, -1)))        
        result = result[1:-1]

        return redirect(url_for("results", res=result))
   

    else:
        return render_template("index.html")


@app.route("/<res>")
def results(res):
    return render_template('results.html', results=res)

if __name__ == "__main__":
    app.run()
    
    


  