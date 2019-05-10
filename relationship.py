
import numpy as np
import matplotlib.pyplot as plt
def getData():
    f = open("load.txt", "r")
    # Remove column headers
    train_data, train_labels, test_data, test_labels = [], [], [], []
    lines = f.readlines()[1:]
    for line in lines:
        line = line.split()
        # element = (time, t_db, RH, Rad, load)
        element = np.array([line[0], line[1], line[2], line[3], line[4]])
        # Convert into floats
        element = np.asarray(element, dtype = float)
        # Add element to data
        train_data.append(element[:4])
        train_labels.append(element[4])
    f.close()
    
    test_data = np.array(train_data[5000:8000])
    test_labels = np.array(train_labels[5000:8000])
    train_data = np.array(train_data[:5000])
    train_labels = np.array(train_labels[:5000])
    print(min(train_labels), max(train_labels))

    return (train_data, train_labels), (test_data, test_labels)

def pltNormalised():
    m0=max([d[0] for d in train_data])
    m1=max([d[1] for d in train_data])
    m2=max([d[2] for d in train_data])
    m3=max([d[3] for d in train_data])
    prefix = 'normalized-'

    plt.scatter([d[0]/m0 for d in train_data], train_labels,s=1,label='load')
    plt.xlabel('Normalised Inputs')
    plt.ylabel('Load')
    plt.scatter([d[1]/m1 for d in train_data], train_labels,s=1, label='t_db')
    plt.scatter([d[2]/m2 for d in train_data], train_labels,s=1, label='RH')
    plt.scatter([d[3]/m3 for d in train_data], train_labels,s=1, label='Rad')
    plt.legend(loc='upper left')
    plt.show()
    #plt.savefig(prefix+'rel.png')

def pltNormalisedTime():
    m0=max([d[0] for d in train_data])
    t = [d[0] for d in train_data]/m0
    m1=max([d[1] for d in train_data])
    m2=max([d[2] for d in train_data])
    m3=max([d[3] for d in train_data])
    ml = max(train_labels)
    print(t)
    prefix = 'time_normalized-'

    plt.xlabel('Time')
    plt.ylabel('Normalised Inputs')
    plt.scatter(t, train_labels/ml, s=1,label='load')
    plt.scatter(t,[d[1]/m1 for d in train_data], s=1, label='t_db')
    plt.scatter(t,[d[2]/m2 for d in train_data], s=1, label='RH')
    plt.scatter(t,[d[3]/m3 for d in train_data], s=1, label='Rad')
    plt.legend(loc='upper left')
    plt.savefig(prefix+'rel.png')

def pltVisual():
    prefix = ''
    plt.scatter([d[0] for d in train_data], train_labels,s=1)
    plt.xlabel('Time')
    plt.ylabel('Load')
    plt.savefig(prefix+'load-time.png')
    plt.clf()
    plt.scatter([d[1] for d in train_data], train_labels,s=1)
    plt.xlabel('t_db')
    plt.ylabel('Load')
    plt.savefig(prefix+'load-t_db.png')
    plt.clf()
    plt.scatter([d[2] for d in train_data], train_labels,s=1)
    plt.xlabel('RH')
    plt.ylabel('Load')
    plt.savefig(prefix+'load-RH.png')
    plt.clf()
    plt.scatter([d[3] for d in train_data], train_labels,s=1)
    plt.xlabel('Rad')
    plt.ylabel('Load')
    plt.savefig(prefix+'load-rad.png')

(train_data, train_labels), (test_data, test_labels) = getData()

pltNormalisedTime()
