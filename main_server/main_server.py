from flask import Flask, request,render_template,redirect
from jinja2 import FileSystemLoader,Environment
import requests, json
import os
import matplotlib.image as mpimg
from PIL import Image
import ast
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
from agg_train import model_aggregation

from Diagonasis import get_prediction,transform_image

app = Flask(__name__)

cwd = os.getcwd()

@app.route('/')
def main():
    return render_template('home.html',page_name="Home DR using FL")

@app.route('/home')
def home():
    return render_template('home.html',page_name="DR using FL")

@app.route('/Diagnosis', methods=['GET', 'POST'])
def Diagnosis():
    if request.method == 'POST':
        if 'DRfile' not in request.files:
            return redirect(request.url)
        file = request.files.get('DRfile')
        if not file:
           return
        img_bytes = file.read()
        file_tensor = transform_image(image_bytes=img_bytes) #######
        class_name = get_prediction(file_tensor)
        return render_template('result.html', page_name="DR result",result=class_name)
    return render_template('index.html',page_name="DR Diagnosis")

@app.route('/Train', methods=['GET', 'POST'])
def Train():
    return render_template('recive.html',page_name="Fed Training")

@app.route('/clientstatus', methods=['GET','POST'])
def client_status():
	url = "http://localhost:8001/serverack"

	if request.method == 'POST':
		client_port = request.json['client_id']
		
		with open(cwd + '/clients.txt', 'a+') as f:
			f.write('http://localhost:' + client_port + '/\n')

		print(client_port)

		if client_port:
			serverack = {'server_ack': '1'}
			return str(serverack)
		else:
			return "Client status not OK!"
	else:
		return "Client GET request received!"
		
@app.route('/cfile', methods=['POST'])
def filename():
	if request.method == 'POST':
		file = request.files['model'].read()
		fname = request.files['json'].read()
		# cli = request.files['id'].read()

		fname = ast.literal_eval(fname.decode("utf-8"))
		cli = fname['id']+'\n'
		fname = fname['fname']
		
		return "<h1>str(fname)</h1>"
		
@app.route('/cmodel', methods=['POST'])
def getmodel():
	if os.path.isdir(cwd + '/client_models') == False:
		os.mkdir(cwd + '/client_models')
	if request.method == 'POST':
		file = request.files['model'].read()
		fname = request.files['json'].read()
		# cli = request.files['id'].read()

		fname = ast.literal_eval(fname.decode("utf-8"))
		cli = fname['id']+'\n'
		fname = fname['fname']

		# with open('clients.txt', 'a+') as f:
		# 	f.write(cli)
		
		print(fname)
		wfile = open(cwd + "/client_models/"+fname, 'wb')
		wfile.write(file)
			
		return "Model received!"
	else:
		return "No file received!"
		
@app.route('/aggregate_models')
def perform_model_aggregation():
	model_aggregation()
	return render_template("agg.html")

@app.route('/send_model_clients')
def send_agg_to_clients():
	try:
		with open(cwd + '/clients.txt', 'r') as f:
			clients = f.read()
		clients = clients.split('\n')
		
		for c in clients:
			if c != '':

				file = open(cwd + "/agg_model/agg_model.pt", 'rb')
				data = {'fname':'agg_model.pt'}
				files = {
					'json': ('json_data', json.dumps(data), 'application/json'),
					'model': ('agg_model.pt', file, 'application/octet-stream')
				}
				print("file readed")
				cli = c+"aggmodel"
				print(cli)
				req = requests.post(url=cli, files=files)
				print("request sent")
				print(req.status_code)
		
		return render_template("sent.html")
	except Exception as e:
		print(e)
		print("server error")




if __name__ == '__main__':
	app.run(host='localhost', port=8000, debug=False, use_reloader=True)
