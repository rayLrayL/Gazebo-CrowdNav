import matplotlib.pyplot as plt
import csv
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='arguments')
parser.add_argument('--name', type=str, default='temp', help='csv file name')
#parser.add_argument('--test', '-t', action='calculate RMSE', help='RMSE values')
args = parser.parse_args()


if __name__ == '__main__':
    with open(args.name+'.csv') as file:
        csv_data = []
        for idx, line in enumerate(file.readlines()):
            if idx ==0 or idx == 1:
                continue
            temp = line.split(', ')
            temp[0] = temp[0][1:]
            temp[-1] = temp[-1][:-2]
            if temp[-1] == '':
                continue
            temp = np.array(temp, dtype = 'float32')
            csv_data.append(temp)
        csv_data = csv_data[:-1]

    temp = []
    for idx, data in enumerate(csv_data):
        if idx in np.arange(0,len(csv_data)+1, 4):
            temp.append(data)
    csv_data = temp

    csv_data = np.array(csv_data)
    time = csv_data[:,0]
    dpoom = [csv_data[:,1], csv_data[:,2]]
    agents = []
    for i in range(5):
        agents.append([csv_data[:,3+i*2], csv_data[:,4+i*2]])

    # plot slip angle Max err
    plt.figure(figsize=(5, 5))
    plt.title(args.name)
    plt.scatter(dpoom[0], dpoom[1], label='dpoom', s=35, linewidth=0.3, color='tab:orange')
    plt.plot(dpoom[0], dpoom[1],color='tab:orange')


    plt.scatter(agents[0][0], agents[0][1], label='agent'+str(1), s=35, alpha=0.8, linewidth=0.3,facecolors='none', edgecolors='tab:green')
    plt.plot(agents[0][0], agents[0][1], color='tab:green', alpha=0.5, linewidth=0.5)
    plt.scatter(agents[1][0], agents[1][1], label='agent'+str(2), s=35, alpha=0.8, linewidth=0.3,facecolors='none', edgecolors='tab:brown')
    plt.plot(agents[1][0], agents[1][1], color='tab:brown', alpha=0.5, linewidth=0.5)
    plt.scatter(agents[2][0], agents[2][1], label='agent'+str(3), s=35, alpha=0.8, linewidth=0.3,facecolors='none', edgecolors='tab:purple')
    plt.plot(agents[2][0], agents[2][1], color='tab:purple', alpha=0.5, linewidth=0.5)
    plt.scatter(agents[3][0], agents[3][1], label='agent'+str(4), s=35, alpha=0.8, linewidth=0.3,facecolors='none', edgecolors='tab:pink')
    plt.plot(agents[3][0], agents[3][1], color='tab:pink', alpha=0.5, linewidth=0.5)
    plt.scatter(agents[4][0], agents[4][1], label='agent'+str(5), s=35, alpha=0.8, linewidth=0.3,facecolors='none', edgecolors='tab:olive')
    plt.plot(agents[4][0], agents[4][1], color='tab:olive', alpha=0.5, linewidth=0.5)

    plt.scatter(0, 3, marker='*', color='r')
    squares = [[-2.5, -3.5, -3.5, 3.5, 2.5, -0.5, -1.5, 2.5], [1.5, -1.5, -2.5, -1.5, -4.5, -5.5, -5.5, -6.5]]
    squares = np.array(squares, dtype='float32')
    plt.scatter(squares[0], squares[1], marker='s', s=570, color='tab:blue')
    plt.scatter(0.5, -1.5, marker='o', s=570, color='tab:blue')
    plt.xlim(-5.7, 5.7)
    plt.ylim(-8.2,3.2)
    plt.xlabel('x(m)', fontsize=8)
    plt.ylabel('y(m)', fontsize=8)

    for idx in range(len(dpoom[0])):
        if idx == len(dpoom[0])-1:
            plt.text(dpoom[0][idx], dpoom[1][idx], "{:.2f}".format(idx/2), fontsize=13)
        if idx in np.arange(0,7, 8):
            plt.text(dpoom[0][idx], dpoom[1][idx], "{:.2f}".format(idx/2), fontsize=13)
    for agent in agents:
        for idx in range(len(agent[0])):
            #if idx == len(agent[0])-1:
                #plt.text(agent[0][idx], agent[1][idx], "{:.2f}".format(idx/2), fontsize=9)
            if idx in np.arange(0,1, 8):
                plt.text(agent[0][idx], agent[1][idx], "{:.2f}".format(idx/2), fontsize=9)
    print(csv_data[0][2])
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.show()
