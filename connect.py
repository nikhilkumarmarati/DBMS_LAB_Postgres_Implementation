import sqlite3

conn = sqlite3.connect('images.db')  # connects or creates a database file
cursor = conn.cursor()

# initializes the database with a table for images if it doesn't exist
# create a table named 'images' with columns id INT PRIMARY KEY,embedding TEXT
cursor.execute('''''
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY,
        embedding TEXT,
    )
    ''')
conn.commit()

# read from features.txt and insert into the database
with open('features.txt', 'r') as f:
    for line in f:
        # split the line into id and embedding
        id, embedding = line.strip().split(',', 1)
        # insert the id and embedding into the database
        cursor.execute('INSERT INTO images (id, embedding) VALUES (?, ?)', (id, embedding))
conn.commit()

#print the database to check if the data is inserted correctly
cursor.execute('SELECT * FROM images')
rows = cursor.fetchall()