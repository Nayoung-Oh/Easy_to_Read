'''
@ Contributor: Nayoung-Oh, darae-lee

'''

import argparse

from sympy import false
from trainer import Trainer
import torch
from preprocess_data import DataProcessor
from scipy import spatial
import csv

# increment index score by inc
def change_scores(scores, index, inc):
  scores[index] = scores[index] + inc
  return scores
# compute difference b/w output and input (nonuse)
def designated_change(index, input_feature, output_feature):
  return output_feature[index]-input_feature[index]
# compute other scores diff (nonuse)
def other_change(index, input_feature, output_feature):
  distance = []
  for i in range(5):
    if i==index:
      distance.append(0.0)
      continue
    distance.append((input_feature[i]-output_feature[i])**2)
  return distance

# return 2 scores
def total_output(data, model, src, dest, index, delta):
  original_features = data.make_features(src, dest)
  original_features = original_features[-5:]
  designated_features = change_scores(original_features.copy(), index, delta)
  new_dest = model.simplify(src, torch.tensor([designated_features], dtype=torch.float))
  output_features = data.make_features(src, new_dest)
  output_features = output_features[-5:]
  return designated_features, output_features

def sim(List1, List2):
    result = 1 - spatial.distance.cosine(List1, List2)
    return round(result, 4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='wikilarge')
    parser.add_argument('--model', type=str, default='feature')
    parser.add_argument('--loss', type=str, default='weighted')
    parser.add_argument('--path', type=str, default='log')
    parser.add_argument('--easse', type=bool, default=False)

    args = parser.parse_args()

    variant = vars(args)

    trainer = Trainer(variant["data"], variant["model"], variant["loss"])
    trainer.transformer.load_state_dict(torch.load(variant["path"]))
    data = DataProcessor(variant["data"])
    data.change_type("test")
    data.generate_data()

    if variant['easse']:
      with open("./"+variant["data"]+"/test.csv", encoding = 'utf-8') as f:
        with open("./report.txt", "w", encoding='utf-8') as res:
            reader = csv.reader(f)
            for l in reader:
                res.write(trainer.simplify(l[-2], torch.tensor([[float(i) for i in l[5:10]]])) + '\n')

    else:
      with open(data.dest_file, 'r') as f1:
          test_dest = f1.readlines()
      with open(data.src_file, 'r') as f2:
          test_src = f2.readlines()
      
      total_len = len(test_src)

      f0_designates = []
      f0_outputs = []
      for i in range(total_len):
        f0_designate, f0_output = total_output(data, trainer, test_src[i],test_dest[i],0,1.0)
        f0_designates.append(f0_designate)
        f0_outputs.append(f0_output)

      f1_designates = []
      f1_outputs = []
      for i in range(total_len):
        f1_designate, f1_output = total_output(data, trainer, test_src[i],test_dest[i],1,1.0)
        f1_designates.append(f1_designate)
        f1_outputs.append(f1_output)

      f2_designates = []
      f2_outputs = []
      for i in range(total_len):
        f2_designate, f2_output = total_output(data, trainer, test_src[i],test_dest[i],2,0.2)
        f2_designates.append(f2_designate)
        f2_outputs.append(f2_output)

      f3_designates = []
      f3_outputs = []
      for i in range(total_len):
        f3_designate, f3_output = total_output(data, trainer, test_src[i],test_dest[i],3,0.2)
        f3_designates.append(f3_designate)
        f3_outputs.append(f3_output)

      f4_designates = []
      f4_outputs = []
      for i in range(total_len):
        f4_designate, f4_output = total_output(data, trainer, test_src[i],test_dest[i],4,0.2)
        f4_designates.append(f4_designate)
        f4_outputs.append(f4_output)

      result0=sim(sum(f0_designates,[]),sum(f0_outputs,[])) # +1
      result1=sim(sum(f1_designates,[]),sum(f1_outputs,[])) # +1
      result2=sim(sum(f2_designates,[]),sum(f2_outputs,[])) # +0.2
      result3=sim(sum(f3_designates,[]),sum(f3_outputs,[])) # +0.2
      result4=sim(sum(f4_designates,[]),sum(f4_outputs,[])) # +0.2
      print(result0, result1, result2, result3, result4)

