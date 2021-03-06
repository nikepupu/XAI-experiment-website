from flask import Flask,render_template,request, make_response, jsonify
import os
import numpy as np
import random
import copy
import time

from pymongo import MongoClient
import uuid
import json


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

@app.route('/exam',methods=['GET', 'POST'])
def exam():
  return render_template("exam.html")
    
@app.route('/start',methods=['POST', 'GET'])
def legends(show=True):

	img1 = ["ThingsInTheCabinet.png"]
	cap1 = ["This is a cabinet which contains some objects: apple, carrots, mango, and orange. The location of the cabinet is marked on the pciture at the top "]

	img2 = ["topdown_view_v1.png"]
	cap2 = ["This is the top down view of the scene, and it marks your position and the position of various important objects."]	

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
  img1 = ["ThingsInTheCabinet.png"]
  cap1 = ["This is a cabinet which contains some objects: apple, carrots, mango, and orange. The location of the cabinet is marked on the pciture at the top "]

  img2 = ["topdown_view_v1.png"]
  cap2 = ["This is the top down view of the scene, and it marks your position and the position of various important objects."]  

  return render_template("example.html", task = "Making Orange Juice", explanation = explanation,images1 = img1, images2 = img2, cap1 = cap1, cap2 = cap2, show = False)


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
