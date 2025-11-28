from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(db.Model):
    id = db.column(db.Integer, primary_key= True)
    user_name = db.column(db.String(156),unique = True, nullable = False)
    password_hash = db.column(db.String(256),nullable = False)

    def set_passowrd(self,password):
        self.password_hash = generate_password_hash(password)

    def check_password(self,password):
        return check_password_hash(self.password_hash,password)