from flask import Flask, jsonify, request, render_template
import json
import numpy as np
import pickle
import joblib

# with open("final_model.pkl", "rb") as f:
#     model = pickle.load(f)
model = joblib.load('final_model.pkl')

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def index():
    pred = ""
    X = []
    if request.method == "POST":
        cust_quantity = request.form["cust_quantity"]
        cust_other_usage = request.form["cust_other_usage"]
        cust_coupon_usage = request.form["cust_coupon_usage"]
        camp_duration = request.form["camp_duration"]
        item_NO = request.form["item_NO"]
        brand_NO = request.form["brand_NO"]
        items_quantity = request.form["items_quantity"]
        X = np.array([[float(cust_quantity), 
                       float(cust_other_usage), 
                       float(cust_coupon_usage),
                       float(camp_duration),
                       float(item_NO),
                       float(brand_NO),
                       float(items_quantity)]])
        pred = model.predict(X)
    return render_template("index.html", pred=pred, X=X)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
