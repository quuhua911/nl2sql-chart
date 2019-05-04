from flask import Flask, request
#from backend import model, prediction
#from backend.dbinfo import db
import backend.dbinfo as dbinfo
import json


def select_db(sql, dbName):
    db = dbinfo.createDB(dbName)
    result = db.engine.execute(sql)
    return result


def add_db(sql):
    return sql
