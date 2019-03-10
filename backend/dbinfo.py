from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import urllib

app = Flask(__name__)

'''
For mssql:

driver = "ODBC Driver 17 for SQL Server"
server = "127.0.0.1"
database = "TestDB"
uid = "sa"
pwd = "Zh010203"

params = urllib.parse.quote_plus("DRIVER={%s};SERVER=%s;DATABASE=%s;UID=%s;PWD=%s"
                                 % (driver, server, database, uid, pwd))

app.config['SQLALCHEMY_DATABASE_URI'] = "mssql+pyodbc:///?odbc_connect=%s" % params
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

'''


def createDB(dbName):
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///../database/' + dbName + "/" + dbName + ".sqlite"
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
    db = SQLAlchemy(app)
    return db
