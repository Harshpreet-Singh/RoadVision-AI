import psycopg2

try:
    conn = psycopg2.connect(
        host="localhost",
        database="roadvision_db",
        user="postgres",
        password="postgres"  # apna password
    )
    print("Connected to PostgreSQL!")
    conn.close()
except Exception as e:
    print("Connection failed:", e)