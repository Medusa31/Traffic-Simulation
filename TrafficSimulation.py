# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 18:00:26 2016

@author: Carl
"""
#keyboard shortcuts:
    #indent/unindent: tab/shift+tab
    #comment/uncomment: ctrl+1
#file created to store code that works
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import random

    
def one_lane_plot(length,cars,v_min,v_max,steps,randomise):
# one lane of traffic
# represented as a one dimensional array of possible sites for cars
# with closed and periodic b.c.'s   
# create unique random sample of car positions and sort in reverse order
# create arrays for lane, initial car position and velocity
    lane = np.zeros(length)     
    carvel = rnd.randint(v_min,v_max,size=cars)  
    rancarpos = np.array([])
    carpos = np.array([])
    q=random.sample(range(len(lane)-1), cars)
    i = 0
    for i in np.arange(cars):       
        lane[q[i]] = 1
        rancarpos = np.append(rancarpos,q[i])
        i = i+1
    for rancarpos in sorted(rancarpos):
        carpos = np.append(carpos, rancarpos)
    i = 0
    xvals = np.array([])
    yvals = np.array([])
    velvals = np.array([])
    #carvel = [4,5,5,5,0,0,1,4,5]
    #carpos = [70,59,51,49,48,40,33,26,9]
    #carpos = [9,26,33,40,48,49,51,59,70]
    flowcount = 0
    for i in np.arange(steps):
        j = 0
        for j in np.arange(cars):
            #print("Position of",j, "is", carpos[j])
            #print("Velocity of",j, "is", carvel[j])         
            lane[carpos[j]%len(lane)] = 0
            k=carpos[j%len(lane)]+carvel[j]
            infront = np.array([])  #array for the space in which cars are in front
            for m in np.arange(carpos[j]+1,k+1,1):
                infront = np.append(infront, lane[m%len(lane)])
                #print("lane:",m," is ",lane[m%len(lane)])
            #print(infront)
            if sum(infront) == 0:
                lane[k%len(lane)] = 1
                carpos[j] = k
                if carvel[j] < v_max and lane[(carpos[j]+1)%len(lane)]==0: #b.c. for acceleration
                    carvel[j]=carvel[j] + 1
                    #carvel[1] = 0
            else:
                count = 0
                while infront[count] == 0:
                    count=count+1
                carvel[j] = count
                lane[(carpos[j]+carvel[j])%len(lane)] = 1
                carpos[j] = carpos[j]+carvel[j]
            if carpos[j] != carpos[j]%len(lane):
                flowcount = flowcount + 1
            carpos[j]=carpos[j]%len(lane)
            j = j + 1
            #randomise velocities
        P=randomise/rnd.random_sample()
        for n in range (cars):
            if carvel[n] > v_min and P > 1:
                carvel[n]=carvel[n]-1
            n=n+1
        l = 0
        i = i + 1
        for l in range(cars):
            xvals = np.append(xvals, carpos[l])
            yvals = np.append(yvals, i)
            velvals = np.append(velvals, carvel[l])
            l = l + 1
    #plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.scatter(xvals,yvals,c=velvals,cmap=plt.cm.hot)
    fig.suptitle('High density traffic modelling on a circular road', fontsize=15)
    ax.set_xlabel('distance', fontsize=15)
    ax.set_ylabel('timestep', fontsize=15)
    plt.xlim([0,length])
    plt.ylim([0,steps])
    fig.savefig('lowdensity.png')
    plt.show()           
    density = cars/length            
    flow = flowcount/(steps)
    print("flow:",flow)
    print("density:",density)
    return [density,flow]
    

def one_lane(length,cars,v_min,v_max,steps,randomise):
# one lane of traffic
# represented as a one dimensional array of possible sites for cars
# with closed and periodic b.c.'s   
# create unique random sample of car positions and sort in reverse order
# create arrays for lane, initial car position and velocity
    lane = np.zeros(length)     
    carvel = rnd.randint(v_min,v_max,size=cars)  
    rancarpos = np.array([])
    carpos = np.array([])
    q=random.sample(range(len(lane)-1), cars)
    i = 0
    for i in np.arange(cars):       
        lane[q[i]] = 1
        rancarpos = np.append(rancarpos,q[i])
        i = i+1
    for rancarpos in sorted(rancarpos):
        carpos = np.append(carpos, rancarpos)
    i = 0
    xvals = np.array([])
    yvals = np.array([])
    velvals = np.array([])
    #carvel = [4,5,5,5,0,0,1,4,5]
    #carpos = [70,59,51,49,48,40,33,26,9]
    #carpos = [9,26,33,40,48,49,51,59,70]
    flowcount = 0
    for i in np.arange(steps):
        j = 0
        for j in np.arange(cars):
            #print("Position of",j, "is", carpos[j])
            #print("Velocity of",j, "is", carvel[j])         
            lane[carpos[j]%len(lane)] = 0
            k=carpos[j%len(lane)]+carvel[j]
            infront = np.array([])  #array for the space in which cars are in front
            for m in np.arange(carpos[j]+1,k+1,1):
                infront = np.append(infront, lane[m%len(lane)])
                #print("lane:",m," is ",lane[m%len(lane)])
            #print(infront)
            if sum(infront) == 0:
                lane[k%len(lane)] = 1
                carpos[j] = k
                if carvel[j] < v_max and lane[(carpos[j]+1)%len(lane)]==0: #b.c. for acceleration
                    carvel[j]=carvel[j] + 1
            else:
                count = 0
                while infront[count] == 0:
                    count=count+1
                carvel[j] = count
                lane[(carpos[j]+carvel[j])%len(lane)] = 1
                carpos[j] = carpos[j]+carvel[j]
            if carpos[j] != carpos[j]%len(lane):
                flowcount = flowcount + 1
            carpos[j]=carpos[j]%len(lane)
            j = j + 1
            #randomise velocities
        P=randomise/rnd.random_sample()
        for n in range (cars):
            if carvel[n] > v_min and P > 1:
                carvel[n]=carvel[n]-1
            n=n+1
        l = 0
        i = i + 1
        for l in range(cars):
            xvals = np.append(xvals, carpos[l])
            yvals = np.append(yvals, i)
            velvals = np.append(velvals, carvel[l])
            l = l + 1
    #plot          
    density = cars/length            
    flow = flowcount/(steps)
    return [density,flow]
    
    
denplot = np.array([])
flowplot = np.array([])
for i in np.arange(1,100,20):
    densityflow = one_lane(100,i,0,5,50,0)
    denplot = np.append(denplot,densityflow[0])
    flowplot = np.append(flowplot,densityflow[1])
print(denplot)
print(flowplot)

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
ax.scatter(denplot,flowplot)
fig.suptitle('Density against flow rate for single lane traffic', fontsize=15)
ax.set_xlabel('Density', fontsize=15)
ax.set_ylabel('Flow rate', fontsize=15)
plt.xlim([0,1])


#one_lane_plot(100,10,0,5,20,0)
#plt_onelane()





#def randomise_vels(p):
#    i=0
#    P=p/rnd.random_sample()
#    for i in range (cars):
#       if carvel[i] > v_min and P > 1:
#           carvel[i]=carvel[i]-1
#    i=i+1
#
#def plt_onelane():
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    fig.subplots_adjust(top=0.85)
#    ax.scatter(xvals,yvals,c=velvals,cmap=plt.cm.hot)
#    fig.suptitle('High density traffic modelling on a circular road', fontsize=15)
#    ax.set_xlabel('distance', fontsize=15)
#    ax.set_ylabel('timestep', fontsize=15)
#    plt.xlim([0,length])
#    plt.ylim([0,steps])
#    #ax.text((length*(7/8))/2, steps+4, r'$\rho=0.6$', fontsize=15,
#    #        bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})
#    fig.savefig('lowdensity.png')
#    plt.show()
#
#
##def randomise_vels(p):
##    i=0
##    P=p/rnd.random_sample()
##    for i in range (cars):
##       if carvel[i] > v_min and P > 1:
##           carvel[i]=carvel[i]-1
##    i=i+1
#
#def plt_onelane():
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    fig.subplots_adjust(top=0.85)
#    ax.scatter(xvals,yvals,c=velvals,cmap=plt.cm.hot)
#    fig.suptitle('High density traffic modelling on a circular road', fontsize=15)
#    ax.set_xlabel('distance', fontsize=15)
#    ax.set_ylabel('timestep', fontsize=15)
#    plt.xlim([0,length])
#    plt.ylim([0,steps])
#    #ax.text((length*(7/8))/2, steps+4, r'$\rho=0.6$', fontsize=15,
#    #        bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})
#    fig.savefig('lowdensity.png')
#    plt.show()


#    lane = np.zeros(length)
#    carvel = rnd.randint(v_min,v_max,size=cars)    
#    rancarpos = np.array([])
#    carpos = np.array([])
#    q=random.sample(range(len(lane)-1), cars)
#    i = 0
#    for i in np.arange(cars):       
#        lane[q[i]] = 1
#        rancarpos = np.append(rancarpos,q[i])
#        i = i+1
#        for rancarpos in sorted(rancarpos, reverse=True):
#            carpos = np.append(carpos, rancarpos)
    #oldcarpos = np.array([])
