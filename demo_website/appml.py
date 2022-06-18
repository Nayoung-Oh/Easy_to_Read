'''
@ Contributor: Nayoung-Oh, ZosiaZamoyska

'''
import argparse
from trainer import Trainer
import torch
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3"

our_transformer = Trainer('wikilarge', 'feature', 'weighted')
baseline = Trainer('wikilarge', 'naive', 'none')
our_transformer.transformer.load_state_dict(torch.load("new_large_6_val_1.046"))
baseline.transformer.load_state_dict(torch.load("baseline_3_val_2.484"))

#########################################################
#########################################################
#################### FLASK CODE #########################
#########################################################
#########################################################

from flask import Flask, render_template, request, send_file
import json
app = Flask(__name__)

@app.route("/")
def hello_world():
    # return render_template("hello.html")
    return send_file("templates/hello.html")


@app.route("/api/askmodel", methods = ['POST'])
def post():
    print(request.get_data())
    print(request.is_json)

    params = json.loads(request.get_data(), encoding='utf-8')
    print(params)

    arr = [
        float(params['feat1']),
        float(params['feat2']),
        float(params['feat3']),
        float(params['feat4']),
        float(params['feat5']),
    ]
    output = our_transformer.simplify(str(params['text']), torch.tensor([arr], dtype=torch.float))
    output_baseline = baseline.simplify(str(params['text']), torch.tensor([arr], dtype=torch.float))
    data = {"output": output, 
    "output-baseline": output_baseline}
    return data

    
if __name__ == '__main__':
      app.run(host='0.0.0.0', port=8888)

