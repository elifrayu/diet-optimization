import mysql.connector

def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",   # şifre koymadıysan boş
        database="diet_db"
    )