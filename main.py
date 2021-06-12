from flask import Flask
from flask import jsonify
from flask import request
machineLearningAlog = Flask(__name__)

@app.route('/', methods=['POST'])
def machineLearningAlog():

    data = request.get_json('data')

    name = data['name']
    email = data['email']
    success=""
    
    if name == "kamal" and email == "kamal@gmail.com":
        success = 'Login Successfull'
    else:
        success = 'Login Failed'


    return jsonify({success:success})