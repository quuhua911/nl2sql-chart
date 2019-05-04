# -*- coding: UTF-8 -*-

from flask import Flask, request, render_template, send_file
from flask_cors import *
from flask_sqlalchemy import SQLAlchemy
from backend import prediction
from backend.dbinfo import app

import os
import json

# app = Flask(__name__)
CORS(app, supports_credentials=True)


# 建表
# db.create_all()

# 添加
# admin = UUser('admin','123@123.com')
# db.session.add(admin)
# db.session.commit()

# 输出


def read_themessage():

    messages = model.Message.query.all()

    for message in messages:
        one_message = json.dumps(message, default=model.obj_to_json)
        return one_message

# me = Message(0,0,'Test Again')
# db.session.add(me)
# db.session.commit()

@app.route('/dbs')
def read_dbs():
    options = []
    for file in os.listdir("./database/"):
        temp = {}
        temp["value"] = file
        temp["label"] = file
        options.append(temp)
    options.sort(key=lambda x: x['value'])
    return json.dumps(options)

@app.route('/')
def read_message():

    user_input = request.args['input']
    db_id = request.args['db']

    seq_valid = prediction.val_seq(user_input)

    if seq_valid:
        result = prediction.get_final_from_seq(user_input, db_id)
    else:
        result = {
            "uflag": 0,
            "content": "Invalid Sentence!",
            "table": '',
            "labels": '',
            "type_of_chart": '',
            "predicted_x_col": '',
            "predicted_y_col": '',
            "xy_data": ''
        }
    print(result)
    return json.dumps(result)


@app.route('/test')
def test():
    return render_template("test.html")


@app.route('/home')
def index():
    return render_template("index.html")



