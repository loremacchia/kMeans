import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

k = 0

plot = "times"


if plot == "scatter":
    dataX = []
    dataY = []
    dataZ = []
    trueColors = []
    kmColors = []
    confrontColors = []
    area = []
    # row_count = sum(1 for row in csv.reader( open('newDataset.csv') ) )
    #
    # with open('newDataset.csv') as File:
    #     i = 0
    #     reader = csv.reader(File, delimiter=',', quotechar=',',
    #                         quoting=csv.QUOTE_MINIMAL)
    #     for row in reader:
    #         if i < row_count:
    #             dataX.append(round(float(row[0]),3))
    #             dataY.append(round(float(row[1]),3))
    #             trueColors.append(float(row[2]))
    #             kmColors.append(not float(row[3])) if float(float(row[3]) != 2) else kmColors.append(float(row[3]))
    #             confrontColors.append('#8e8e8e') if trueColors[i] == kmColors[i] else confrontColors.append('#ff0000')
    #             area.append(2)
    #             i=i+1
    #
    row_count = sum(1 for row in csv.reader( open('newDataset3D.csv') ) )

    with open('newDataset3D.csv') as File:
        i = 0
        reader = csv.reader(File, delimiter=',', quotechar=',',
                            quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            if i < row_count:
                dataX.append(round(float(row[0]),3))
                dataY.append(round(float(row[1]),3))
                dataZ.append(round(float(row[2]),3))
                trueColors.append(float(row[3]))
                # kmColors.append(not float(row[3])) if float(float(row[3]) != 2) else kmColors.append(float(row[3]))
                kmColors.append(float(row[4]))
                area.append(5)
                i=i+1


    print(len(dataX))

    x = dataX
    y = dataY
    z = dataZ

    # fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    # fig.suptitle('Horizontally stacked subplots')
    # ax1.scatter(x, y, s=area, c=trueColors, alpha=0.5)
    # ax2.scatter(x, y, s=area, c=kmColors, alpha=0.5)
    # ax3.scatter(x, y, s=area, c=confrontColors, alpha=0.5)

    fig = pyplot.figure()
    ax2 = Axes3D(fig)
    ax2.scatter(x, y, z, s=area, c=kmColors, alpha=0.5)

elif plot == "times":
    dataX = []
    dataYOMP = []
    dataYCPP = []
    # dataYCUDA1 = []
    # dataYCUDA2 = []
    # dataYCUDA3 = []

    with open('omp.csv') as File:
        reader = csv.reader(File, delimiter=',', quotechar=',',
                            quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            dataX.append(row[0])
            dataYOMP.append(round(float(row[1]),6))

    with open('cpp.csv') as File:
        reader = csv.reader(File, delimiter=',', quotechar=',',
                            quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            dataYCPP.append(round(float(row[1]),6))

    # with open('cuda1.csv') as File:
    #     reader = csv.reader(File, delimiter=',', quotechar=',',
    #                         quoting=csv.QUOTE_MINIMAL)
    #     for row in reader:
    #         dataYCUDA1.append(round(float(row[1]),6))
    #
    # with open('cuda2.csv') as File:
    #     reader = csv.reader(File, delimiter=',', quotechar=',',
    #                         quoting=csv.QUOTE_MINIMAL)
    #     for row in reader:
    #         dataYCUDA2.append(round(float(row[1]),6))
    #
    # with open('cuda3.csv') as File:
    #     reader = csv.reader(File, delimiter=',', quotechar=',',
    #                         quoting=csv.QUOTE_MINIMAL)
    #     for row in reader:
    #         dataYCUDA3.append(round(float(row[1]),6))



    fig, ax = plt.subplots()
    ax.plot(dataX, dataYCPP, '-b', label='CPP')
    ax.plot(dataX, dataYOMP, '-r', label='OMP')
    # ax.plot(dataX, dataYCUDA1, '-y', label='CUDA1')
    # ax.plot(dataX, dataYCUDA2, '-g', label='CUDA2')
    # ax.plot(dataX, dataYCUDA3, '-p', label='CUDA3')
    plt.xlabel('#points')
    plt.ylabel('execution time (s)')
    leg = ax.legend()

    for i in range(3):
        dataX.pop()
        dataYOMP.pop()
        dataYCPP.pop()

    fig, ax = plt.subplots()
    ax.plot(dataX, dataYCPP, '-b', label='CPP')
    ax.plot(dataX, dataYOMP, '-r', label='OMP')
    # ax.plot(dataX, dataYCUDA1, '-y', label='CUDA1')
    # ax.plot(dataX, dataYCUDA2, '-g', label='CUDA2')
    # ax.plot(dataX, dataYCUDA3, '-p', label='CUDA3')
    plt.xlabel('#points')
    plt.ylabel('execution time (s)')
    leg = ax.legend()






plt.show()