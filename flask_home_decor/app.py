from flask import Flask, abort, render_template, jsonify, request
from api import get_input, find_product

#from api import find_book
import numpy as np
import pandas as pd
import pickle
import dill


app = Flask('predict')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    d = {'result':'error'}
    if request.method == 'POST':
        if 'file' not in request.files:
            d['result'] = 'failure'
            return jsonify(d)
        file = request.files['file']
        file.save("/Users/user/Documents/Data_Science/Projects/flask_home_decor/user_image")
        d['result'] = 'success'
        print("values: ")
        print(request.form.get('description'))
        response =find_product(request.form.get('description'))
        return jsonify(response)



@app.route('/')
def index():
    return render_template('index.html')

app.run()
