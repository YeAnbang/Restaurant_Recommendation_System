#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
#import geopandas as gpd
import re
import sklearn
import numpy as np
from sklearn.neighbors import NearestNeighbors
import socket
import sys
import threading
import json
import numpy as np

# In[2]:
import multiprocessing



#sample data format
Q = [[0, 0, 2, 0.0329, -78.6, 10000.0, 9.0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
       0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0,
       0.2222222222222222]]

to_send = "#AFSDG,"+" ".join([str(i) for i in Q[0]])+"@"
to_send_encoded = to_send.encode("utf-8")
print(to_send_encoded)


def sender():
    # read user_vec from client, put into cosumer queue
    # que obj support internal blocking mechanism, maybe don't need a lock
    print("setting client sender")
    host = socket.gethostname()
    port = 12349
    port2 = 12348

    
    #client side
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.connect((host, port2))
    myaddr = serversocket.getsockname()
    print("server addr:%s" % str(myaddr))
    while True:
        #print("client addr:",str(address)," server address:",str(serversocket.getsockname()))
        serversocket.send(to_send_encoded)    #client send test instance to server
        serversocket.send(to_send_encoded)    #client send test instance to server
        serversocket.send(to_send_encoded)    #client send test instance to server
        serversocket.send(to_send_encoded)    #client send test instance to server
        print("send sucess")
        
        


def writer():
    # read out writer queue send back results to client
    # return binary encoded str, b'#2#AFSDG,259 43 243@'
    #worker_id#token,recomendation1 recomendation2 recomendation3
    host = socket.gethostname()
    port = 12349
    port2 = 12348
    
    #server side
    testsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    testsocket.bind((host, port))
    testsocket.listen()
    
    
    #client side
    
    testsocket.settimeout(10)
    t, address = testsocket.accept()
    print("set writer")
    #print("writer server addr:%s" % str(myaddr))
    while True:
        tmp = t.recv(1024)    #server accept test instance from server
        if len(tmp) == 0:
            continue
        print("test client receive: ",tmp)   #--> should output test client receive:  b'#AFSDG,259 43 243@'
            
    return 

'''
pool = multiprocessing.Pool(processes=2)
m = multiprocessing.Manager()
process1 = pool.apply_async(sender)
process2 = pool.apply_async(writer)
pool.close()
pool.join()
'''
sender()

# In[ ]:





# In[ ]:




