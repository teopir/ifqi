import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
from variableLoadSave import ExperimentVariables
import numpy as np
import argparse
from matplotlib2tikz import save as tikz_save
from matplotlib.backends.backend_pdf import PdfPages

font = {'family' : 'normal',
        'size'   : 22,
        'style' : "normal"}

font_label = {'family' : 'normal',
        'size'   : 27,
        'style' : "normal"}
mpl.rc('font', **font)

parser = argparse.ArgumentParser(
    description='Plot3D')

parser.add_argument("experimentName",type=str, help="Provide the name of the experiment")
parser.add_argument("var",type=str, help="Provide the name of the variable to plot")
parser.add_argument("max_iter", type=int, help="Provide the max number of iteration")
parser.add_argument("max_regr", type=int, help="Provide the max number of complexity")
#parser.add_argument("regrOpt",type=int, help="Provide regressior option")
parser.add_argument("-s", "--std", help="Standardization", action="store_true")
parser.add_argument("-r", "--rev", help="1 - x", action="store_true")
parser.add_argument("-l", "--log", help="Score is logarithmic", action="store_true")
parser.add_argument("-o", "--loss", help="Plot Loss", action="store_true")
parser.add_argument("-j", "--jump", help="Jump every five iterations", action="store_true")
parser.add_argument("-m", "--mlp", help="Jump every five iterations", action="store_true")
parser.add_argument("-x", "--latex", help="Jump every five iterations", action="store_true")
parser.add_argument("-c", "--color", help="Colored plots", action="store_true")
parser.add_argument("-t", "--ten", help="ten or twenty", action="store_true")
parser.add_argument("-tl", "--topleft", help="topleft legend", action="store_true")

args = parser.parse_args()
experimentName = args.experimentName
max_iter = args.max_iter
max_regr = args.max_regr
regrChoiche = 0#args.regrOpt
varName = args.var
isLog = args.log
isStd = args.std
isRev = args.rev
hasLoss = args.loss
jump = args.jump
is_mlp = args.mlp
latex = args.latex
color = args.color
ten = args.ten
top_left = args.topleft

hmap_kargs = {"cmap":"Greys"}
nameAdd = ""
if color:
    hmap_kargs = {}
    nameAdd = "_color"

if is_mlp:
    nameAdd += "MLP"
    lastTitle = "Neural Network"
else:
    nameAdd += "EXTRA"
    lastTitle = "Extra Trees"
regr_cut = 0

expVar = ExperimentVariables(experimentName)

iterations_list = sorted(expVar.iterationLoaded)
if jump:
    iterations_list = [x for x in expVar.iterationLoaded if x==1 or x%5==0]

diffMat = np.zeros((len(expVar.regressorLoaded)/2,len(iterations_list)))
ensMat = np.zeros((len(expVar.regressorLoaded)/2,len(iterations_list)))
sinMat = np.zeros((len(expVar.regressorLoaded)/2,len(iterations_list)))
sLossMat = np.zeros((len(expVar.regressorLoaded)/2,len(iterations_list)))
sValLossMat = np.zeros((len(expVar.regressorLoaded)/2,len(iterations_list)))
sNEpochMat = np.zeros((len(expVar.regressorLoaded)/2,len(iterations_list)))
eLossMat = np.zeros((len(expVar.regressorLoaded)/2,len(iterations_list)))
eValLossMat = np.zeros((len(expVar.regressorLoaded)/2,len(iterations_list)))
eNEpochMat = np.zeros((len(expVar.regressorLoaded)/2,len(iterations_list)))

X2D = np.zeros((len(expVar.regressorLoaded)/2,len(iterations_list)))
Y2D = np.zeros((len(expVar.regressorLoaded)/2,len(iterations_list)))
Z2D = np.zeros((len(expVar.regressorLoaded)/2,len(iterations_list)))
Z2D1 = np.zeros((len(expVar.regressorLoaded)/2,len(iterations_list)))
Z2D2 = np.zeros((len(expVar.regressorLoaded)/2,len(iterations_list)))
id=0
x=0

if regrChoiche == 0:
    regr_option = range(1,40) #[regr_cut:]
elif regrChoiche == 1:
    regr_option = [5,10,15,20,25,50,75,100,200,500,750,
               1000,1500,2000,3000,4000,5000,7500,
               10000,12500,15000,20000,25000,30000,35000,40000,45000,50000,
               55000,60000,65000,70000,75000,80000,85000,90000,95000,100000]#[regr_cut:]



section_single = [0]*(len(expVar.regressorLoaded)/2)
section_ensemble = [0]*(len(expVar.regressorLoaded)/2)
section_diff = [0]*(len(expVar.regressorLoaded)/2)

section_single_std = [0]*(len(expVar.regressorLoaded)/2)
section_ensemble_std = [0]*(len(expVar.regressorLoaded)/2)
section_diff_std = [0]*(len(expVar.regressorLoaded)/2)

min_plt = np.infty
max_plt = -np.infty
#Standardization
min_v = np.infty
max_v = -np.infty
min_diff_v = np.infty
max_diff_v = -np.infty

for regressor in range(len(expVar.regressorLoaded)/2):#[:-regr_cut-1]:
    y=0
    for iteration in sorted(iterations_list):

        single_score = expVar.load((regressor+regr_cut)*2,0,iteration,varName)[0]
        ensemble_score = expVar.load((regressor+regr_cut)*2+1,0,iteration,varName)[0]

        min_diff_v = min(min_diff_v,ensemble_score-single_score)
        max_diff_v = max(max_diff_v, ensemble_score - single_score)
        min_v = min(min_v,single_score,ensemble_score)
        max_v = max(max_v,single_score,ensemble_score)

#regr_option = range(20)

#--------------------------------------------------------------------------------------
# Prende i dati e li inserisce in matrici
#--------------------------------------------------------------------------------------

print len(regr_option)
for regressor in range(len(expVar.regressorLoaded)/2):#[:-regr_cut-1]:
    y=0
    for iteration in sorted(iterations_list):
        if hasLoss:
            sLossMat[x,y]  = expVar.load((regressor+regr_cut)*2,0,iteration,"loss")[0]
            sValLossMat[x, y] = expVar.load((regressor+regr_cut)*2,0,iteration,"valLoss")[0]
            sNEpochMat[x,y] = expVar.load((regressor+regr_cut)*2,0,iteration,"nEpoch")[0]
            eLossMat[x,y] = expVar.load((regressor + regr_cut) * 2+1, 0, iteration, "loss")[0]
            eValLossMat[x, y] = expVar.load((regressor + regr_cut) * 2 + 1, 0, iteration, "valLoss")[0]
            eNEpochMat[x,y] = expVar.load((regressor + regr_cut) * 2 + 1, 0, iteration, "nEpoch")[0]

        single_score = expVar.load((regressor+regr_cut)*2,0,iteration,varName)[0]
        ensemble_score = expVar.load((regressor+regr_cut)*2+1,0,iteration,varName)[0]
        diff_score = ensemble_score - single_score



        if isStd:
            single_score = (single_score - min_v) / (max_v - min_v) * 0.9 + 0.05
            ensemble_score = (ensemble_score - min_v) / (max_v - min_v)* 0.9 + 0.05
            diff_score = (diff_score - min_diff_v) / (max_diff_v - min_diff_v)* 0.9 + 0.05

        if isLog:
            if isRev:
                single_score = -np.log(1-single_score)
                ensemble_score = -np.log(1-ensemble_score)
                diff_score = -np.log(1-diff_score)
            else:
                single_score = np.log(single_score)
                ensemble_score = np.log(ensemble_score)
                diff_score = np.log(diff_score)


        X2D[x,y] = regr_option[regressor]
        Y2D[x,y] = iteration
        Z2D[x,y] = diffMat[x,y] = ensemble_score-single_score#diff_score

        Z2D1[x,y] = ensMat[x,y] = ensemble_score
        Z2D2[x,y] = sinMat[x,y] = single_score

        max_plt = max(max_plt,ensemble_score, single_score)
        min_plt = min(min_plt, ensemble_score, single_score)

        y+=1
    x+=1
    s = expVar.load((regressor+regr_cut) * 2, 0, iterations_list[max_iter], varName)
    f = expVar.load((regressor+regr_cut) * 2 + 1, 0,iterations_list[max_iter], varName)
    section_single[regressor] = s[0]
    section_ensemble[regressor] = f[0]
    section_diff[regressor] = section_ensemble[regressor] - section_single[regressor]
    section_single_std[regressor] = s[1] / np.sqrt(s[2]) * 1.96
    section_ensemble_std[regressor] = f[1] / np.sqrt(f[2]) * 1.96
    section_diff_std[regressor] = section_ensemble_std[regressor] + section_single_std[regressor]

print X2D#
print Y2D
print Z2D
fig = plt.figure()

last_iter_indx = None if max_iter < 0 else max_iter + 1
max_regr = None if max_regr< 0 else max_regr + 1
X2D = X2D[:max_regr,:last_iter_indx]
Y2D = Y2D[:max_regr,:last_iter_indx]
Z2D = Z2D[:max_regr,:last_iter_indx]
ensMat = ensMat[:max_regr,:last_iter_indx]
sinMat = sinMat[:max_regr,:last_iter_indx]
diffMat = diffMat[:max_regr,:last_iter_indx]
Z2D1 = Z2D1[:max_regr,:last_iter_indx]
Z2D2 = Z2D2[:max_regr,:last_iter_indx]
if not hasLoss:
    filepath = directory = "plot3D/" + experimentName + "/" + varName
    #print "PATH", filepath
    #directory = os.path.dirname(filepath)
    if not os.path.isdir(directory): os.makedirs(directory)

    """
    ax = fig.gca(projection='3d')
    if color:
        ax.plot_surface(X2D, Y2D, Z2D,rstride=1,cstride=1,color='b')
    else:
        ax.plot_surface(X2D, Y2D, Z2D,rstride=1,cstride=1,color='0.75')
    plt.savefig(filepath + "/3DDiff" + nameAdd + ".jpg")
    pp = PdfPages(filepath + "/3DDiff" + nameAdd + ".pdf")
    plt.savefig(pp, format='pdf',bbox_inches='tight')
    pp.close()
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    if color:
        ax.plot_surface(X2D, Y2D, Z2D1,rstride=1,cstride=1,color='g')
    else:
        ax.plot_surface(X2D, Y2D, Z2D1,rstride=1,cstride=1,color='0.75')
    plt.savefig(filepath + "/3DB-FQI" + nameAdd + ".jpg")
    pp = PdfPages(filepath + "/3DB-FQI" + nameAdd + ".pdf")
    plt.savefig(pp, format='pdf',bbox_inches='tight')
    pp.close()
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    if color:
        ax.plot_surface(X2D, Y2D, Z2D2,rstride=1,cstride=1,color='r')
    else:
        ax.plot_surface(X2D, Y2D, Z2D2,rstride=1,cstride=1,color='0.75')
    plt.savefig(filepath + "/3DFQI" + nameAdd + ".jpg")
    pp = PdfPages(filepath + "/3DFQI" + nameAdd + ".pdf")
    plt.savefig(pp, format='pdf',bbox_inches='tight')
    pp.close()
    plt.show()
    """

    x_label = "max depth"
    if is_mlp:
        x_label = "#neurons"
    y_label = "#iterations"
    #Set the labels and the ticks to their proper values and positions
    x_n_ticks = 5
    ticks = np.array([1,5,10,15,20])
    ticks = np.array([1,5,10,15,20,25,30,35,40])
    x_dist = sinMat.shape[0]/x_n_ticks
    x_new_labels = range(1,sinMat.shape[0],x_dist)
    y_n_ticks = 5
    y_dist = sinMat.shape[1] / y_n_ticks
    y_new_labels = [sorted(iterations_list)[x] for x in range(0, sinMat.shape[1], y_dist)]

    #HeatMaps
    ax = plt.subplot(1, 1, 1)
    ax.xaxis.set(ticks=ticks-0.5, ticklabels=ticks)
    ax.yaxis.set(ticks=np.arange(0.5, len(y_new_labels)*y_dist,y_dist), ticklabels=y_new_labels)
    plt.title("B-FQI")
    heatmap = plt.pcolor(ensMat.T,vmin=min_plt,vmax=max_plt,**hmap_kargs)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.colorbar(heatmap)
    #plt.savefig(filepath + "/B-FQI" + nameAdd + ".jpg")
    pp = PdfPages(filepath + "/B-FQI" + nameAdd + ".pdf")
    plt.savefig(pp, format='pdf',bbox_inches='tight')
    pp.close()
    if not latex:
        plt._show()
    else:
        tikz_save(filepath + '/FQI' + nameAdd + '++.tex')
    plt.clf()

    ax = plt.subplot(1, 1, 1)
    ax.xaxis.set(ticks=ticks-0.5, ticklabels=ticks)
    ax.yaxis.set(ticks=np.arange(0.5, len(y_new_labels)*y_dist,y_dist), ticklabels=y_new_labels)
    plt.title("FQI")
    heatmap = plt.pcolor(sinMat.T,vmin=min_plt,vmax=max_plt,**hmap_kargs)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.colorbar(heatmap)
    #plt.savefig(filepath + "/FQI" + nameAdd + ".jpg")
    pp = PdfPages(filepath + "/FQI" + nameAdd + ".pdf")
    plt.savefig(pp, format='pdf', bbox_inches='tight')
    pp.close()
    if not latex:
        plt._show()
    else:
        tikz_save(filepath + '/FQI' + nameAdd + '.tex')
    plt.clf()

    ax = plt.subplot(1,1,1)
    ax.xaxis.set(ticks=ticks-0.5, ticklabels=ticks)
    ax.yaxis.set(ticks=np.arange(0.5, len(y_new_labels)*y_dist,y_dist), ticklabels=y_new_labels)
    plt.title("Difference")
    heatmap = plt.pcolor(diffMat.T, **hmap_kargs)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.colorbar(heatmap)
    #plt.savefig(filepath + "/FQIDiff" + nameAdd + ".jpg")
    pp = PdfPages(filepath + "/FQIDiff" + nameAdd + ".pdf")
    plt.savefig(pp, format='pdf', bbox_inches='tight')
    pp.close()
    if not latex:
        plt._show()
    else:
        tikz_save(filepath + '/Diff' + nameAdd + '.tex')
    plt.clf()

    def multiOneDimPlot(data, xlabel, ylabel, title, path):
        global  font_label, top_left
        color = ["r", "g", "b", "k"]
        dashes = [":", "--"]
        imgName = os.path.realpath(path)
        i = 0
        hand = []
        min_x = np.inf
        min_ = np.inf
        max_ = -np.inf
        max_x = -np.inf
        sort_keys = reversed(sorted(data))
        for dic in sort_keys:
            dict_ = data[dic]
            name = dict_["name"]
            mean = dict_["mean"]
            conf = dict_["conf"]

            if max(dict_["iteration"]) > max_x:
                max_x = max(dict_["iteration"])
            if min(dict_["iteration"]) < min_x:
                min_x = min(dict_["iteration"])
            if min(mean) < min_:
                min_ = min(mean)
            if max(mean) > max_:
                max_ = max(mean)

            if (len(conf) != 0):
                plt.errorbar(dict_["iteration"], mean, color=color[i], yerr=conf, lw=2, label=name , ls=dashes[len(hand)])
            temp = plt.plot(dict_["iteration"], mean, color=color[i], lw=2, label=name, ls=dashes[len(hand)])
            hand.append(temp[0])

            i += 1
            i %= len(color)

        #plt.rc('text', usetex=True)
        #plt.rc('font', family='serif')
        l = plt.ylabel(ylabel, fontdict= font_label)
        plt.xlabel(xlabel)
        plt.title(title)
        loc = 4
        if top_left:
            loc = 0
        plt.legend(handles=hand, loc=loc)
        boundaries = max(abs(min_) * 0.1, 0.1 * abs(max_))
        plt.ylim(min_ - boundaries, max_ + boundaries)
        plt.xlim(min_x * 0.95, max_x * 1.05 )
        #plt.savefig(imgName)
        pp = PdfPages(imgName.split('.')[0] + ".pdf")
        plt.savefig(pp, format='pdf', bbox_inches='tight')
        pp.close()
        if not latex:
            plt._show()

    data = {"B-FQI":{"name":"B-FQI", "mean":section_ensemble[:max_regr], "conf":section_ensemble_std[:max_regr], "iteration":regr_option[:max_regr]},#len(section_ensemble)]},
            "fqi": {"name": "FQI", "mean": section_single[:max_regr], "conf": section_single_std[:max_regr], "iteration":regr_option[:max_regr]}}#len(section_ensemble)]}}

    if ten:
        val = "10"
    else:
        val = "20"
    nomeY = varName if not varName=="score" else r"$J^{\pi_{" + val + "}}$"
    multiOneDimPlot(data,x_label, nomeY,lastTitle,"plot3D/" + experimentName + "/" + varName + "/Last" + nameAdd + ".jpg")
    if latex:
        tikz_save(filepath + '/last_section.tex')


    data = {"diff":{"name":"Difference", "mean":section_diff[:max_regr], "conf":section_diff_std[:max_regr], "iteration":regr_option[:max_regr]}}#len(section_ensemble)]}}

    #multiOneDimPlot(data,"regressor", "score","Last Iteration","last_section_diff.jpg")

else:
    ax = fig.gca(projection='3d')
    ax.plot_surface(X2D, Y2D, sLossMat, rstride=1, cstride=1, color='b')
    plt.savefig("plot3D/" + experimentName + "/" + varName + "/FQI.jpg")
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X2D, Y2D, sValLossMat, rstride=1, cstride=1, color='g')
    plt.savefig("Q.jpg")
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X2D, Y2D, sNEpochMat, rstride=1, cstride=1, color='r')
    plt.savefig("Q.jpg")
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X2D, Y2D, eLossMat, rstride=1, cstride=1, color='b')
    plt.savefig("Q.jpg")
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X2D, Y2D, eValLossMat, rstride=1, cstride=1, color='g')
    plt.savefig("Q.jpg")
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X2D, Y2D, eNEpochMat, rstride=1, cstride=1, color='r')
    plt.savefig("Q.jpg")
    plt.show()