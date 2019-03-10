#from backend.dbinfo import db


class Message(db.Model):

    __tablename__ = 'Message'
    # key
    id = db.Column(db.Integer,primary_key = True)
    # mark the group
    groupID = db.Column(db.Integer)
    # represent the client type
    uflag = db.Column(db.Integer)
    # 0: None
    # 1: Line
    # 2: Histogram
    # 3: Pie
    type_of_chart = db.Column(db.Integer)
    # mark the x-y col
    x_col = db.Column(db.String(100))
    y_col = db.Column(db.String(100))
    # content resultJSON
    content = db.Column(db.String(100))

    def __init__(self, id, groupID, uflag, type_of_chart, x_col, y_col, content):
        self.id = id
        self.groupID = groupID
        self.uflag = uflag
        self.type_of_chart = type_of_chart
        self.x_col = x_col
        self.y_col = y_col
        self.content = content

    def __repr__(self):
        return '<Message Group:%r>' % self.id


def obj_to_json(obj):
    return {
        "id": obj.id,
        "groupID": obj.groupID,
        "uflag": obj.uflag,
        "type_of_chart": obj.type_of_chart,
        "x_col": obj.x_col,
        "y_col": obj.y_col,
        "content": obj.content
    }

