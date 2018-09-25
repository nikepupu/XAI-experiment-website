from flask import Flask,render_template,request
import os
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
	print next_i
	print next_j

	title, action, exp1 , exp2, img , imgtitle, ans1, ans2 = training_phase(int(next_j), int(next_i))
	next_j = str(int(next_j) +  (int(next_i) + 1)/4)
	next_i = str( (int(next_i)+1)%4 )
	if int(next_j) < 2:
		return render_template("example.html", title = title, action = action, exp1 = exp1 , exp2 = exp2, img = img , imgtitle = imgtitle
		, ans1 = ans1, ans2 = ans2,next_i = next_i, next_j = next_j)
	else:
		return "hello world"


if __name__ == '__main__':
    app.run(debug = True)
