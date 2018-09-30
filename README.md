database setup guide:

start mongodb server:

sudo service mongod start

go to mongodb shell:

(first time use only)

use XAI

db.createCollection("userProgress")

When deploying on aws or gcp etc need to change the last line of code  for app.py

app.run(host = '0.0.0.0', port = 80) # 80 is the port for http

ToDo:

Setup browser cookie to track user progress
Change the entire testing phase code to javascript to avoid doing refrequent and unnecessary update to the database and avoiding global set to hold robot instance for each user.




