# -*- coding: UTF-8 -*-

from flask import Flask, request
from flask_cors import *
from flask_sqlalchemy import SQLAlchemy
#from backend import model, prediction
from backend import prediction
from backend.dbinfo import app
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


@app.route('/')
def read_message():
    # open file.
    #input_file = open("./testData.json", 'w')
    #user_input = request.args['test']
    #input_file.write(user_input + '\n')
    #result = prediction.predict_sql("./testData.json")

    user_input = request.args['input']
    db_id = request.args['db']

    db_id = "concert_singer"
    # sql = prediction.predict_sql
    sql = "select * from singer"

    # for test
    the_bar_input = "Show name, country, age for all singers ordered by age from the oldest to the youngest."
    the_line_input = "For each stadium, how many concerts play there?"

    # (x_col, y_col, type) = prediction.predict_charts
    x_col = 1
    y_col = 5
    types = 0

    if user_input == '1':
        types = 1
    elif user_input == '2':
        types = 2
    elif user_input == the_bar_input:
        types = 1
        x_col = 0
        y_col = 2
        sql = "SELECT name ,  country ,  age FROM singer ORDER BY age DESC"
    elif user_input == the_line_input:
        types = 2
        x_col = 0
        y_col = 1
        sql = "SELECT T2.name ,  count(*) as stats FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id GROUP BY T1.stadium_id"
    result = prediction.get_final_from_seq(user_input, db_id, sql, x_col, y_col, types)
    print(result)
    return json.dumps(result)
    # return '{"uflag":0,"groupID":1,"content":"Test"}' + request.args['test']


def for_test():
    user_input = "123"
    db_id = "concert_singer"
    # sql = prediction.predict_sql
    sql = "select * from singer"

    # (x_col, y_col, type) = prediction.predict_charts
    x_col = 1
    y_col = 5
    types = 1

    result = prediction.get_final_from_seq(user_input, db_id, sql, x_col, y_col, types)
    print(result)
    return json.dumps(result)


if __name__ == '__main__':

    # test = for_test()
    app.run()

    # resultEnd = prediction.predict_sql("./testData.json")

    #predicted_sql = prediction.predict_sql_from_seq("How many singers do we have?", "concert_singer")

    # all for debug
    # debug1 = prediction.predict_sql("./testData.json")