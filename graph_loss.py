import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from string_globals import *
import os
import csv

def data_to_csv(name,model_type,data):
    save_dir='{}/{}'.format(graph_dir,name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open('{}/{}.csv'.format(save_dir,model_type), 'a+', newline='') as csvfile:
        wrtr=csv.writer(csvfile)
        wrtr.writerows([ [d] for d in data])


def line_graph(name,model_type,data):
    x=[_ for _ in range(len(data))]
    plt.ylim(min(data),max(data))
    plt.plot(x,data)
    plt.title(model_type)
    save_dir='{}/{}'.format(graph_dir,name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path='{}/{}'.format(save_dir,model_type)
    i=0
    while os.path.exists(save_path):
        i+=1
        save_path='{}/{}_{}'.format(save_dir,model_type,i)
    plt.savefig(save_path)
    plt.show()

if __name__ == '__main__':
    data_to_csv('testing_csv','auto',[_ for _ in range(5)])
    data_to_csv('testing_csv','auto',[_ for _ in range(10,20,1)])