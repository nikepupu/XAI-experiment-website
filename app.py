from flask import Flask,render_template,request, make_response, jsonify
import os
import numpy as np
import random
import copy
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pymongo import MongoClient
import uuid
import json

from robot import robot

app = Flask(__name__, static_url_path='/static')
app.jinja_env.filters['zip'] = zip

#bots = set()

client = MongoClient()
db = client.XAI
doc = db['userProgress']

exp = 0
noexp = 0

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
            allFiles.append(entry)
                
    return allFiles 

	

def new_experiment():

	return exp1 , exp2, img , imgtitle, sub_task_idx,a

@app.route('/',methods=['GET', 'POST'])
def index():
  return render_template("intro.html")
    
@app.route('/start',methods=['POST', 'GET'])
def legends(show=True):

	img1 = ["Cabinet.png", "ThingsInTheCabinet.png"]
	cap1 = ["cabinet", "Things in the cabinet"]

	img2 = ["topdown_label.png"]
	cap2 = ["top down view of the scene"]	

	return render_template("legends.html", images1 = img1, images2 = img2, cap1 = cap1, cap2 = cap2, show = show)


@app.route('/hints', methods=['POST', 'GET'])
def hints():
	return legends(show = False)

@app.route('/example',  methods=['POST', 'GET'])
def example():
	global exp, noexp
	explanation = True
	if exp <= noexp:
		explanation = True
		exp += 1
	else:
		explanation = False
		noexp += 1

	return render_template("example.html", task = "Making Orange Juice", explanation = explanation)

@app.route('/result',  methods=['POST', 'GET'])
def result():
	res = request.json
	
	while True:
   		idnum = str(uuid.uuid4())
   		cursor = doc.find({idnum: {"$exists": True}}).limit(1)
   		if cursor.count() == 0:
   			break

   	data={
   	"user_id_xai":idnum,
   	"response": res

   	}
   	key = {"user_id_xai":idnum}
   	doc.update(key, data, upsert=True)
   	with open("result.txt", 'a') as outfile:
   		json.dump(data, outfile)
   	
	return jsonify(status="success")

@app.route('/final',  methods=['POST', 'GET'])
def final():
	return render_template("result.html")

if __name__ == '__main__':
    app.run(debug = False)
