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


customer_array = np.load("customer_array.npy",allow_pickle=True)
customer_rare_array = np.load("customer_rare_array.npy",allow_pickle=True)
by_customer_rare = pd.read_csv('./by_customer_rare.csv')
by_customer = pd.read_csv('./by_customer.csv')
customer_rare = pd.read_csv('./customer_rare.csv')
customer = pd.read_csv('./customer.csv')


# In[3]:


'''
samples = [[0, 0, 2], [1, 0, 0], [0, 0, 1]]
neigh = NearestNeighbors(2, 0.4)
neigh.fit(samples) 
neigh.kneighbors([[0, 0, 1.3]], 2, return_distance=False)
'''
neigh = NearestNeighbors()
neigh.fit(customer_array) 

neigh_rare = NearestNeighbors()
neigh_rare.fit(customer_rare_array) 


# In[4]:


by_customer = by_customer.set_index('customer_id')
by_customer_rare = by_customer_rare.set_index('customer_id')


# In[5]:


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


# In[6]:



def query(user_vecs, neigh, table, pool, k_value=3, return_num=1):
    dist, knn = neigh.kneighbors(user_vecs, k_value, return_distance=True)
    recommendation = []
    recommendation_weight = []
    #print(dist,knn)
    for k,i in enumerate(list(knn)):
        
        #print(pool.iloc[i])
        user_id = table.iloc[i]['customer_id'].values
        recommended_restaurant = []
        for id in user_id:
            recommended_restaurant.append(pool.loc[id]['VENDOR'].split(' '))
        confidence = {}
        for j in range(len(recommended_restaurant)):
            for restaurant in recommended_restaurant[j]:
                if restaurant not in confidence:
                    confidence[restaurant]=0.
                confidence[restaurant]+=1./dist[0][j]
        p = np.asarray(list(confidence.values()))
        p/=np.sum(p)
        keys = list(confidence.keys())
        recommendation.append(np.random.choice(len(confidence.keys()), return_num, p=p,replace=False))
        ret = [keys[recommendation[-1][n]]for n in range(return_num)]
        #print("generate recommendation for %d'th input..."%k)
        #print("--Recommendation is restaurant: "+str(ret))
        return ret
        

def generate_recommendation(user_vecs):
    l1 = query(user_vecs, neigh, customer, by_customer, k_value=3, return_num=2)
    l2 = query(user_vecs, neigh_rare, customer_rare, by_customer_rare, k_value=3, return_num=1)
    l1.extend(l2)
    return l1

def worker(worker_id,que1,que2):
    #read cosumer queue put result to writer queue
    customer_array = np.load("customer_array.npy",allow_pickle=True)
    customer_rare_array = np.load("customer_rare_array.npy",allow_pickle=True)
    by_customer_rare = pd.read_csv('./by_customer_rare.csv')
    by_customer = pd.read_csv('./by_customer.csv')
    customer_rare = pd.read_csv('./customer_rare.csv')
    customer = pd.read_csv('./customer.csv')
    neigh = NearestNeighbors()
    neigh.fit(customer_array) 
    
    neigh_rare = NearestNeighbors()
    neigh_rare.fit(customer_rare_array) 
    by_customer = by_customer.set_index('customer_id')
    by_customer_rare = by_customer_rare.set_index('customer_id')
    #sample data format
    Q = [[0, 0, 2, 0.0329, -78.6, 10000.0, 9.0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
           0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0,
           0.2222222222222222]]
    
    to_send = "#AFSDG,"+" ".join([str(i) for i in Q[0]])+"@"
    to_send_encoded = to_send.encode("utf-8")
    
    def query(user_vecs, neigh, table, pool, k_value=3, return_num=1):
        dist, knn = neigh.kneighbors(user_vecs, k_value, return_distance=True)
        recommendation = []
        recommendation_weight = []
        #print(dist,knn)
        for k,i in enumerate(list(knn)):
            
            #print(pool.iloc[i])
            user_id = table.iloc[i]['customer_id'].values
            recommended_restaurant = []
            for id in user_id:
                recommended_restaurant.append(pool.loc[id]['VENDOR'].split(' '))
            confidence = {}
            for j in range(len(recommended_restaurant)):
                for restaurant in recommended_restaurant[j]:
                    if restaurant not in confidence:
                        confidence[restaurant]=0.
                    confidence[restaurant]+=1./dist[0][j]
            p = np.asarray(list(confidence.values()))
            p/=np.sum(p)
            keys = list(confidence.keys())
            recommendation.append(np.random.choice(len(confidence.keys()), return_num, p=p,replace=False))
            ret = [keys[recommendation[-1][n]]for n in range(return_num)]
            #print("generate recommendation for %d'th input..."%k)
            #print("--Recommendation is restaurant: "+str(ret))
            return ret
        
    def generate_recommendation(user_vecs):
        l1 = query(user_vecs, neigh, customer, by_customer, k_value=3, return_num=2)
        l2 = query(user_vecs, neigh_rare, customer_rare, by_customer_rare, k_value=3, return_num=1)
        l1.extend(l2)
        return l1
    print("add worker",worker_id)
    while True:
        task = que1.get()
        if task:
            #print("worker %d handle tasks"%worker_id)
            recommendation = generate_recommendation([task[0][1]])
            #print("worker %d handle tasks"%worker_id, recommendation)
            que2.put([(str(worker_id).encode("utf-8")+b'#'+task[0][0],recommendation)])


generate_recommendation(Q)


# In[ ]:





# In[7]:


import multiprocessing

    
    
def reader(que1, que2):
    # read user_vec from client, put into cosumer queue
    # que obj support internal blocking mechanism, maybe don't need a lock
    print("setting server reader")
    host = socket.gethostname()
    port = 12349
    port2 = 12348
    
    #server side
    clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clientsocket.bind((host, port2))
    clientsocket.listen()
    
    recv_data = b""
    
    #client side
    #serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #serversocket.connect((host, port2))
    #myaddr = serversocket.getsockname()
    clientsocket.settimeout(10)
    #print("server addr:%s" % str(myaddr))
    t, address = clientsocket.accept()
    while True:
        #print("client addr:",str(address)," server address:",str(serversocket.getsockname()))
        #serversocket.send(to_send_encoded)    #client send test instance to server
        #serversocket.send(to_send_encoded)    #client send test instance to server
        #serversocket.send(to_send_encoded)    #client send test instance to server
        #serversocket.send(to_send_encoded)    #client send test instance to server
        tmp = t.recv(1024)    #server accept test instance from server
        if len(tmp) == 0:
            raise Exception()
        #print("get data",len(tmp))
        recv_data += tmp
        l = recv_data.split(b'@')
        recv_data = l[-1]
        for i in range(len(l)-1):
            token = l[i][1:7]
            raw_data = l[i][7:].split(b' ')
            #print(raw_data)
            raw_data = [float(k) for k in raw_data]
            que1.put([(token,raw_data)])
        
            


# In[ ]:





# In[ ]:


def writer(que1,que2):
    # read out writer queue send back results to client
    # return binary encoded str, b'#2#AFSDG,259 43 243@'
    #worker_id#token,recomendation1 recomendation2 recomendation3
    host = socket.gethostname()
    port = 12349
    port2 = 12348
    
    #server side
    #testsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #testsocket.bind((host, port))
    #testsocket.listen()
    
    
    #client side
    clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clientsocket.connect((host, port))
    myaddr = clientsocket.getsockname()
    #testsocket.settimeout(10)
    #t, address = testsocket.accept()
    print("writer server addr:%s" % str(myaddr))
    while True:
        task = que2.get()
        if task:
            #print("writer write result")
            to_send_encoded = b'#'+task[0][0]+' '.join(task[0][1]).encode("utf-8")+b'@'
            print(to_send_encoded)
            clientsocket.send(to_send_encoded)  
            #tmp = t.recv(1024)    #server accept test instance from server
            #if len(tmp) == 0:
            #    continue
            #print("test client receive: ",tmp)   #--> should output test client receive:  b'#AFSDG,259 43 243@'
            
    return 

pool = multiprocessing.Pool(processes=12)
m = multiprocessing.Manager()
q1 = m.Queue()
q2 = m.Queue()
workers = []
process1 = pool.apply_async(reader, (q1,q2))
process2 = pool.apply_async(worker, (1,q1,q2))
process3 = pool.apply_async(worker, (2,q1,q2))
process4 = pool.apply_async(worker, (3,q1,q2))
process5 = pool.apply_async(worker, (4,q1,q2))
process6 = pool.apply_async(worker, (5,q1,q2))
process7 = pool.apply_async(writer, (q1,q2))
pool.close()
pool.join()


# In[ ]:





# In[ ]:




