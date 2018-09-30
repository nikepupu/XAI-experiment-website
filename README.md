database setup guide:

start mongodb server:

sudo service mongod start

go to mongodb shell:

use XAI

db.createCollection("userProgress")

ToDo:

Setup browser cookie to track user progress
Change the entire testing phase code to javascript to avoid doing refrequent and unnecessary update to the database and avoiding global set to hold robot instance for each user.

