import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from string_globals import *
import os
import csv

def data_to_csv(name,model_type,data):
    '''saves data to csv file, where each line is the loss for the epoch

    Parameters
    ----------
    name -- str. name of model like "rennaissance_block1_conv1"
    model_type. --str. type of model (auto, gen or disc)
    data -- [int]. new data to be added to this csv file
    '''
    save_dir='{}/{}'.format(graph_dir,name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open('{}/{}.csv'.format(save_dir,model_type), 'a+', newline='') as csvfile:
        wrtr=csv.writer(csvfile)
        wrtr.writerow(data)


def line_graph(name,model_type,data):
    '''creates time series graph of data over time

    Parameters
    ----------
    name -- str. name of model like "rennaissance_block1_conv1"
    model_type. --str. type of model (auto, gen or disc)
    data -- [int]. new data to be added to this csv file
    '''
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