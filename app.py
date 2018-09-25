from flask import Flask,render_template,request
import os
import numpy as np
import random
import copy
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from robot import robot

app = Flask(__name__, static_url_path='/static')

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
            allFiles.append(entry)
                
    return allFiles 

	


def training_phase(j, i):

			action_num = 2
			instructions = ["action: grab lemon", "action: place cup"]
			explanations = [
		{	
			# whether there is any problem
			True: {
				# whether the problem is related to obs/dec module
				"obs":{True:"lemon in hand", False:"apple in hand"}, \
				"dec":{
					True:"preconditon: hand empty; \n  posteffect: apple in hand", \
					False:"preconditon: hand empty; \n  posteffect: lemon in hand"
					}
			},
			False: {
				"obs":{False:"lemon in hand"}, \
				"dec":{
					False:"preconditon: hand empty; \n  posteffect: lemon in hand"
					}			
			}
		},
		{
			True: {
				"obs":{True:"cup in front of juicer", False:"cup next to juicer"}, \
				"dec":{
					True:"preconditon: cup in hand; \n  posteffect: cup next to juicer", \
					False:"preconditon: cup in hand; \n  posteffect: cup in front of juicer"
					}
			},
			False: {
				"obs":{False:"cup in front of juicer"}, \
				"dec":{
					False:"preconditon: cup in hand; \n  posteffect: cup in front of juicer"
					}			
			}
		}
	]
			img_names = [
					{True: 'hand_apple.png', False: 'hand_lemon.png'},
					{True: 'cup_next_to_juicer.png', False: 'cup_under_juicer.png'},
					]
			ObsErr = [False, True, False, True]
			DecErr = [False, False, True, True]

			title = "Task: Making lemon juice"
			action =  instructions[j]
			ObsExp = explanations[j][ObsErr[i] or DecErr[i]]["obs"][ObsErr[i]]
			DecExp = explanations[j][ObsErr[i] or DecErr[i]]["dec"][DecErr[i]]


			exp =  "Explanations:"
			#exp = exp + " State (vision module output): "
			exp1 =  ObsExp
			#exp = exp + " Task plan (planning module output):"
			exp2 =  DecExp

			img = img_names[j][ObsErr[i] or DecErr[i]]
			
			imgtitle = "Snapshot from robot's camera after the action"
			
			

			if ObsErr[i] == False:
				ans1  = "Correct Answer: y. Because robot's state and snapshot are aligned."
			else:
				ans1 =  "Correct Answer: n. Because robot's state and snapshot do not match."

	

			if DecErr[i] == False:
				ans2  = "Correct Answer: y. Because robot's plan is necessary for the task."
			else:
				ans2 =  "Correct Answer: n. Because robot's plan is irrelavant to the task."
			return title, action, exp1 , exp2, img , imgtitle,ans1, ans2


def user_study():

	task_length = 6
	episode_num = 5

	# f = open("result.txt", "w")
	r1 = robot(T=task_length*episode_num)
	prior_list_obs = []
	prior_list_dec = []
	acc_obs = []
	acc_dec = []
	u_obs = []
	u_dec = []

	instructions = ["action: grab carrot", "action: place pan on stove", "action: turn on the stove",\
		"action: grab cup", "action: place cup under coffee maker", "action: push brew button and get coffee"]
	explanations = [
		{
			True: {
				"obs":{True:"carrot in hand", False:"orange in hand"}, \
				"dec":{
					True:"preconditon: hand empty; \n  posteffect: orange in hand", \
					False:"preconditon: hand empty; \n  posteffect: carrot in hand"
					}
			},
			False: {
				"obs":{False:"carrot in hand"}, \
				"dec":{
					False:"preconditon: hand empty; \n  posteffect: carrot in hand"
					}			
			}
		},
		{
			True: {
				"obs":{True:"pan on stove", False:"pan next to stove"}, \
				"dec":{
					True:"preconditon: pan in hand; \n  posteffect: pan next to stove", \
					False:"preconditon: pan in hand; \n  posteffect: pan on stove"
					}
			},
			False: {
				"obs":{False:"pan on stove"}, \
				"dec":{
					False:"preconditon: pan in hand; \n  posteffect: pan on stove"
					}			
			}
		},
		{
			True: {
				"obs":{True:"hand near stove switch", False:"hand near oven switch"}, \
				"dec":{
					True:"preconditon: oven close; \n  posteffect: oven open", \
					False:"preconditon: stove close; \n  posteffect: stove open"
					}
			},
			False: {
				"obs":{False:"hand near stove switch"}, \
				"dec":{
					False:"preconditon: stove close; \n  posteffect: stove open"
					}			
			}
		},
		{
			True: {
				"obs":{True:"cup in hand", False:"plate in hand"}, \
				"dec":{
					True:"preconditon: hand empty; \n  posteffect: plate in hand", \
					False:"preconditon: hand empty; \n  posteffect: cup in hand"
					}
			},
			False: {
				"obs":{False:"cup in hand"}, \
				"dec":{
					False:"preconditon: hand empty; \n  posteffect: cup in hand"
					}	
			}
		}, 
		{
			True: {
				"obs":{True:"cup under coffee maker", False:"cup next to coffee maker"}, \
				"dec":{
					True:"preconditon: cup in hand; \n  posteffect: cup next to coffee maker", \
					False:"preconditon: cup in hand; \n  posteffect: cup under coffee maker"
					}
			},
			False: {
				"obs":{False:"cup under coffee maker"}, \
				"dec":{
					False:"preconditon: cup in hand; \n  posteffect: cup under coffee maker"
					}
			}
		}, 
		{
			True: {
				"obs":{True:"coffee maker on, cup under coffee maker", \
					False:"coffee maker off, cup under coffee maker"}, \
				"dec":{
					True:"preconditon: cup empty, cup under coffee maker, coffee maker off; \n  posteffect: cup filled with coffee, cup under coffee maker, coffee maker off", \
					False:"preconditon: cup empty, cup under coffee maker, coffee maker on; \n  posteffect: cup filled with coffee, cup under coffee maker, coffee maker on"
					}
			},
			False: {
				"obs":{False:"coffee maker on, cup under coffee maker"}, \
				"dec":{
					False:"preconditon: cup empty, cup under coffee maker, coffee maker on; \n  posteffect: cup filled with coffee, cup under coffee maker, coffee maker on"
					}
			}
		}


	]

	img_names = [
		{True: 'images/hand_orange.PNG', False: 'images/hand_carrot.PNG'},
		{True: 'images/not_on_stove2.PNG', False: 'images/on_stove.PNG'},
		{True: 'images/open_oven.PNG', False: 'images/open_stove.PNG'},
		{True: 'images/hand_plate.PNG', False: 'images/hand_cup.PNG'},
		{True: 'images/cup_not_under_maker.PNG', False: 'images/cup_under_maker.PNG'},
		{True: 'images/push_pour_button_off.PNG', False: 'images/push_pour_button_on.PNG'}
		]

	img_instructions = ["images/maker_off.png", "images/maker_on.png"]
	img0 = mpimg.imread(img_instructions[0])
	img1 = mpimg.imread(img_instructions[1])
	f, axarr = plt.subplots(1,2)
	axarr[0].imshow(img0)
	axarr[0].set_title("coffee maker off")
	axarr[1].imshow(img1)
	axarr[1].set_title("coffee maker on")
	plt.show(block=False)
	raw_input("Showing states of a coffee maker. Press any button to continue.\n")
	plt.close('all')

	# img = mpimg.imread('images/hand_carrot.PNG')
	# plt.imshow(img)
	# plt.show(block=False)
	# time.sleep(2.0)
	# plt.close('all')

	sub_task_length = [3,6]
	task_name = ["cook carrot", "get coffee"]
	num_exp = []


	for j in range(episode_num):
		acc_count_obs = 0
		acc_count_dec = 0
		u_count_obs = 0
		u_count_dec = 0
		sub_task_idx = 0
		for t in range(task_length):
			if t < sub_task_length[sub_task_idx]:
				print "Task: "+ task_name[sub_task_idx]
			else:
				sub_task_idx += 1
				print "Task: "+ task_name[sub_task_idx] 

			a = r1.sample_action()
			expl = r1.generate_exp()

			ErrFlag = False
			for key in a:
				if key == True:
					ErrFlag = True
					break

			ObsFlag = False
			DecFlag = False
			if "obs" in expl.expl_form:
				ObsFlag = True
			if "dec" in expl.expl_form:
				DecFlag = True

			ObsExp = ""
			DecExp = ""
			if ObsFlag:
				ObsExp += explanations[t][ErrFlag]["obs"][a[0]]
			else:
				ObsExp = "No explanation"
			if DecFlag:
				DecExp += explanations[t][ErrFlag]["dec"][a[1]]
			else:
				DecExp = "No explanation"

			print instructions[t]
			print "Explanations: "
			print " State: "
			print "  "+ ObsExp
			print " Task plan: "
			print "  " + DecExp

			resp = []

			img = mpimg.imread(img_names[t][ErrFlag])
			plt.imshow(img)
			plt.title("Snapshot from robot's camera after the action")
			plt.show(block=False)
			res1 = raw_input("\n Do you think the robot's vision module is working fine (y/n)? ")
			res2 = raw_input("Do you think the robot's task plan is right (y/n)? ")
			plt.close('all')

			if "n" in res1:
				resp.append("obs")
			else:
				resp.append("ok")

			if "n" in res2:
				resp.append("dec")
			else:
				resp.append("ok")

			if a[0] == True and resp[0] == "obs":
				acc_count_obs += 1
			elif a[0] == False and resp[0] == "ok":
				acc_count_obs += 1

			if a[1] == True and resp[1] == "dec":
				acc_count_dec += 1
			elif a[1] == False and resp[1] == "ok":
				acc_count_dec += 1

			if resp[0] == "obs":
				u_count_obs += 1

			if resp[1] == "dec":
				u_count_dec += 1

			print "\n"
			print "\n"

			r1.update_mem(resp)

			prior = r1.update_prior()

		num_exp.append(r1.exp_num*1.0/(task_length))
		r1.exp_num = 0

		# average prediction result
		u_obs.append(1.0*u_count_obs/task_length)
		u_dec.append(1.0*u_count_dec/task_length)

		# human prediction accuracy
		acc_obs.append(1.0*acc_count_obs/task_length)
		acc_dec.append(1.0*acc_count_dec/task_length)

		# machine prior
		prior_list_obs.append(prior[0]["obs"])
		prior_list_dec.append(prior[1]["dec"])

	print "Please send back the following data and three figures (with title \"robot prior\", \"user prediction accuracy\" and \"user prediction result\")."
	print "avg_prior_obs", prior_list_obs, "avg_prior_dec", prior_list_dec
	print "prediction_obs", np.mean(u_obs), "prediction_dec", np.mean(u_dec)
	print "acc_obs", np.mean(acc_obs), "acc_dec", np.mean(acc_dec)
	print "explanation number", num_exp

	plt.title("robot prior")
	plt.plot(prior_list_obs)
	plt.plot(prior_list_dec)
	plt.legend(["obs", "dec"])
	plt.show()

	plt.title("user prediction accuracy")
	plt.plot(acc_obs)
	plt.plot(acc_dec)
	plt.legend(["obs", "dec"])
	plt.show()

	plt.title("user prediction result")
	plt.plot(u_obs)
	plt.plot(u_dec)
	plt.legend(["obs", "dec"])
	plt.show()


@app.route('/')
def index():
    return render_template("intro.html")

@app.route('/start')
def legends():
	img = getListOfFiles('static/img')
	sorted(img)
	return render_template("legends.html", images = img)



@app.route('/example',  methods=['GET', 'POST'])
def example():
	form = request.form
	if 'q1' in form:
		q1 = form['q1']
	if 'q2' in form:
		q2 = form['q2']
	if 'next_i' in form and 'next_j' in form:
		next_j = form["next_j"]
		next_i = form["next_i"]
	else:
		next_i = "0"
		next_j = "0"
	title, action, exp1 , exp2, img , imgtitle, ans1, ans2 = training_phase(int(next_j), int(next_i))
	next_j = str(int(next_j) +  (int(next_i) + 1)/4)
	next_i = str( (int(next_i)+1)%4 )
	if int(next_j) < 2:
		return render_template("example.html", title = title, action = action, exp1 = exp1 , exp2 = exp2, img = img , imgtitle = imgtitle
		, ans1 = ans1, ans2 = ans2,next_i = next_i, next_j = next_j)
	else:
		return render_template("attention.html")


if __name__ == '__main__':
    app.run(debug = True)
