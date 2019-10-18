import numpy as np
import logging
import pickle
import time
import importlib
import copy


import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8, 8)

import matplotlib as mpl
from matplotlib import rcParams
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from matplotlib import ticker
from matplotlib import colors
from matplotlib.patches import Ellipse
from matplotlib import gridspec


from scripts import likelihood

from scripts.utils import get_logger

logger = get_logger(level=logging.INFO)


def getStd(inList, step):
    """ Group jets into subgroups and get std"""
    Njets = len(inList)
    stdList = inList[0:Njets - (Njets % step)]

    avgList = []
    for i in range(0, len(stdList), step):
        avgList.append(np.average(stdList[i:i + step]))

    Listavg = np.std(avgList)

    return Listavg


def statSigma(inList):
    return np.std(inList) / np.sqrt(len(inList))


def deltaRootCut(start, end, in_truth_Dic, in_Greedy_Dic, in_BSO_Dic, Width=0):
    truthDic = copy.copy(in_truth_Dic)
    GreedyDic = copy.copy(in_Greedy_Dic)
    BSODic = copy.copy(in_BSO_Dic)

    tot_truthJets = []
    tot_GreedyJets = []
    tot_BSJets = []

    M_hard = truthDic["jetsList"][0][0]["M_Hard"]

    for k in range(end-start):
        truthJets = []
        GreedyJets = []
        BSJets = []
        for i_jet in range(len(truthDic["jetsList"][k])):

            if (
                    (M_hard / 2 - Width < truthDic["jetsList"][k][i_jet]["deltas"][0] < M_hard / 2 + Width)
                    and (M_hard / 2 - Width < GreedyDic["jetsList"][k][i_jet]["deltas"][0] < M_hard / 2 + Width)
                    and (M_hard / 2 - Width < BSODic["jetsList"][k][i_jet]["deltas"][0] < M_hard / 2 + Width)
            ):
                truthJets.append(truthDic["jetsList"][k][i_jet])
                GreedyJets.append(GreedyDic["jetsList"][k][i_jet])
                BSJets.append(BSODic['jetsList'][k][i_jet])

        tot_truthJets.append(truthJets)
        tot_GreedyJets.append(GreedyJets)
        tot_BSJets.append(BSJets)

    truthDic["jetsList"] = tot_truthJets
    GreedyDic["jetsList"] = tot_GreedyJets
    BSODic["jetsList"] = tot_BSJets

    return truthDic, GreedyDic, BSODic


def jetsLogLH(start, end, Dic):

    dic = {}
    Total_jetsListLogLH = []
    avg_logLH = []
    jetsList = []

    for k in range(end-start):

        # if len(Dic["jetsList"][k])>1:
        #     print("k=",k)
        jetsListLogLH = [np.sum(jet["logLH"]) for jet in Dic["jetsList"][k] ]

        jetsList += Dic["jetsList"][k]
        Total_jetsListLogLH+=jetsListLogLH

        if (k+1)%50==0:
            avg_logLH.append(np.average(np.asarray(Total_jetsListLogLH[k-49:k+1]).flatten()))

    """ Standard deviation for the average log LH for the N runs"""
    sigma = np.std(avg_logLH)


    """ Statistical error for the mean log LH for the  total number of jets as err = sqrt(s)/ sqrt(N), where sigma s the sample variance"""
    statSigma = np.std(Total_jetsListLogLH) / np.sqrt(len(Total_jetsListLogLH))

    dic["jetsList"] = np.asarray(jetsList)
    dic["jetsListLogLH"] = Total_jetsListLogLH
    dic["avgLogLH"] = np.asarray(avg_logLH)
    dic["sigma"] = sigma
    dic["statSigma"] = statSigma

    return dic



def deltaRoot(DicList):
    """ Delta root (Jet Invariant Mass in the Toy Model) """

    jetDic = copy.copy(DicList)

    deltaRoot = []
    for jet in jetDic["jetsList"].flatten():
        root_id = jet["root_id"]
        idL = jet["tree"][root_id][0]
        idR = jet["tree"][root_id][1]
        pL = jet["content"][idL]
        pR = jet["content"][idR]

        deltaRoot.append(likelihood.get_delta_LR(pL, pR))

    jetDic["deltaRoot"] = deltaRoot

    return jetDic


def subjetPt(DicList):
    jetDic = copy.copy(DicList)

    pyMin = []
    pyMax = []
    pyDiff = []
    for jet in jetDic["jetsList"].flatten():
        root_id = jet["root_id"]
        pP = jet["content"][root_id]
        idL = jet["tree"][root_id][0]
        idR = jet["tree"][root_id][1]
        pL = jet["content"][idL]
        pR = jet["content"][idR]

        pyMin.append(np.minimum(pL[0], pR[0]))
        pyMax.append(np.maximum(pL[0], pR[0]))

        pyDiff.append(- pL[0] + jet["content"][root_id][0])

    jetDic["SubjetPyMin"] = pyMin
    jetDic["SubjetPyMax"] = pyMax
    jetDic["SubjetPyDiff"] = pyDiff

    return jetDic


def subjetPhi(DicList):
    """ Get dela_root phi angle and subjets angle"""

    jetDic = copy.copy(DicList)

    phiList = []
    phiDeltaList = []

    for jet in jetDic["jetsList"].flatten():

        PhiJet = np.arctan2(jet["content"][jet["root_id"]][0], jet["content"][jet["root_id"]][1])

        root_id = jet["root_id"]
        idL = jet["tree"][root_id][0]
        idR = jet["tree"][root_id][1]
        pL = jet["content"][idL]
        pR = jet["content"][idR]

        delta_vec = (pR - pL) / 2

        """ arctan2 to find the right quadrant"""
        phiDelta = abs(np.arctan2(delta_vec[0], delta_vec[1]) - PhiJet)

        phiDeltaList.append(phiDelta)

        phi1 = np.arctan2(pL[0], pL[1])
        phi2 = np.arctan2(pR[0], pR[1])

        phiList.append(phi1)
        phiList.append(phi2)

    """ Subjets angle"""
    jetDic["SubjetPhi"] = phiList

    """ Delta_root Phi value"""
    phiDeltaList = [entry if entry <= np.pi else (2*np.pi - entry) for entry in phiDeltaList]
    phiDeltaList = [entry if entry <= np.pi/2 else (np.pi - entry) for entry in phiDeltaList]
    jetDic["SubjetPhiDelta"] = phiDeltaList

    return jetDic







"##########"
""" log LH vs dij and theta angles"""


def scanJets(DicList, angles=False, dijmetrics=False):

    """ JetsPhiDelta: Polar angle for Delta
      JetsPhiDeltaRel: Angle for Delta with respect to the jet axis """
    JetsConstPhi = []
    JetsPhiDelta = []
    JetsPhiDeltaRel = []

    dij = []
    dijSubjets = []

    jetDic = copy.copy(DicList)

    jetDic["jetsList"] = np.asarray(jetDic["jetsList"]).flatten()

    for jet in jetDic["jetsList"]:

        # PhiJet = np.arctan2(jet["content"][jet["root_id"]][0], jet["content"][jet["root_id"]][1])

        if dijmetrics:
            """ dij vs logLH"""
            dij = dij + jet["dij"]
            dijSubjets = dijSubjets + [jet["dij"][0]]

        if angles:
            """ Angular quantities"""
            ConstPhi, PhiDelta, PhiDeltaListRel = traversePhi(jet, jet["root_id"], [], [],[])
            jet["ConstPhi"] = ConstPhi
            jet["PhiDelta"] = PhiDelta
            jet["PhiDeltaRel"] = PhiDeltaListRel
            # jet["PhiDeltaRel"]= np.asarray(PhiDelta) - PhiJet

            JetsConstPhi = JetsConstPhi + ConstPhi
            JetsPhiDelta = JetsPhiDelta + PhiDelta
            JetsPhiDeltaRel = np.concatenate((JetsPhiDeltaRel,jet["PhiDeltaRel"]))

    if angles:
        jetDic["JetsConstPhi"] = JetsConstPhi
        jetDic["JetsPhiDelta"] = JetsPhiDelta


        JetsPhiDeltaRel = [entry if entry <= np.pi else (2 * np.pi - entry) for entry in JetsPhiDeltaRel]
        JetsPhiDeltaRel = [entry if entry <= np.pi / 2 else (np.pi - entry) for entry in JetsPhiDeltaRel]
        jetDic["JetsPhiDeltaRel"] = JetsPhiDeltaRel


    if dijmetrics:
        jetDic["dijs"] = np.transpose(dij)
        jetDic["dijSubjets"] = np.transpose(dijSubjets)

    return jetDic





def traversePhi(jet, node_id, constPhiList, PhiDeltaList, PhiDeltaListRel):
    """
    Recursive function that traverses the tree. Gets leaves angle phi, and delta_parent phi angle for all parents in the tree.
    """

    if jet["tree"][node_id, 0] == -1:

        constPhi = np.arctan2(jet["content"][node_id][0], jet["content"][node_id][1])
        constPhiList.append(constPhi)

    else:

        """ Get angle for the splitting value Delta """
        idL = jet["tree"][node_id][0]
        idR = jet["tree"][node_id][1]
        pL = jet["content"][idL]
        pR = jet["content"][idR]

        delta_vec = (pR - pL) / 2

        """ Find subjet angle"""
        PhiPseudoJet = np.arctan2(jet["content"][node_id][0], jet["content"][node_id][1])

        """ arctan2 to find the right quadrant"""
        TempDeltaPhi = np.arctan2(delta_vec[0], delta_vec[1])
        PhiDeltaList.append(TempDeltaPhi)

        PhiDeltaListRel.append( abs(TempDeltaPhi - PhiPseudoJet))

        traversePhi(
            jet,
            jet["tree"][node_id, 0],
            constPhiList,
            PhiDeltaList,
            PhiDeltaListRel,
        )

        traversePhi(
            jet,
            jet["tree"][node_id, 1],
            constPhiList,
            PhiDeltaList,
            PhiDeltaListRel,
        )

    return constPhiList, PhiDeltaList, PhiDeltaListRel




def scanAngles(DicList):
    JetsConstPhi = []
    JetsPhiDelta = []
    JetsPhiDeltaRel =[]

    for jet in DicList["jetsList"].flatten():

        JetsConstPhi = JetsConstPhi + jet["ConstPhi"]
        JetsPhiDelta = JetsPhiDelta + jet["PhiDelta"]
        JetsPhiDeltaRel = np.concatenate((JetsPhiDeltaRel,jet["PhiDeltaRel"]))

    JetsPhiDeltaRel = [entry if entry>=0 else (entry+np.pi) for entry in JetsPhiDeltaRel]
    DicList["JetsConstPhi"] = JetsConstPhi
    DicList["JetsPhiDelta"] = JetsPhiDelta
    DicList["JetsPhiDeltaRel"] = JetsPhiDeltaRel

    return DicList



def scanDij(DicList):

    dij = []
    dijSubjets = []

    for jet in DicList["jetsList"].flatten():

        """ dij vs logLH"""
        dij = dij + jet["dij"]
        dijSubjets = dijSubjets + [jet["dij"][0]]


    DicList["dijs"] = np.transpose(dij)
    DicList["dijSubjets"] = np.transpose(dijSubjets)

    return DicList

"""#######################################################"""
""" Constituents imbalance"""


def scanTreeImbalance(DicList,w=None, startLevel = None):
    """ Get tree and subjet imbalance lists"""
    subjetsImb = []
    treesImb = []

    for jet in DicList["jetsList"].flatten():

        subjetsImb.append(subjetConst(jet, jet["root_id"]))
        treesImb.append(treeImbalance(jet , jet["root_id"],w, startLevel))


    DicList["subjetsImb"] = subjetsImb
    DicList["treesImb"] = treesImb

    return DicList



def treeImbalance(jet, node_id, w, node_idLevel):
    """ Get:
    inners_list: Number of inner nodes (not counting the last splittings where both children are leaves)
    Imbalance: imbalance of a tree
    """
    inners_list, Imbalance = mainTraverse(jet, node_id, [], [], node_idLevel, w=w)

    # if len(inners_list)>0:
    treeIm = sum(Imbalance) / len(inners_list)
    # treeIm = sum(Imbalance)

    return treeIm


def mainTraverse(jet, node_id, inners_list, Imbalance, Nlevel, w=None):
    """
    Recursive function to get a list of the imbalance for all the branches of a tree (at all levels).
    We weight the contribution of each level by a decaying exponential with rate w * the level number Nlevel
    """
    #     inners_list.append(1)

    Imbalance.append(np.exp(- w * Nlevel) * subjetConst(jet, node_id))

    Nlevel += 1
    #     print("Nlevel = ", Nlevel)

    if jet["tree"][node_id, 0] != -1:

        mainTraverse(
            jet,
            jet["tree"][node_id, 0],
            inners_list,
            Imbalance,
            Nlevel,
            w,
        )

        mainTraverse(
            jet,
            jet["tree"][node_id, 1],
            inners_list,
            Imbalance,
            Nlevel,
            w,
        )

        # Add all the inner nodes, except the ones whose both children are leaves.
        # L = jet["tree"][node_id, 0]
        # R = jet["tree"][node_id, 1]
        #
        # if jet["tree"][L, 0] != -1 and jet["tree"][R, 0] != -1:
        inners_list.append(1)

    return inners_list, Imbalance


def subjetConst(jet, node_id):
    """
    Given a parent node, get the imbalance in the number of constituents of the left and right branches
    :param jet:
    :param node_id:
    :return:
    """
    LConst = traverse(jet, jet["tree"][node_id][0], [], [])
    RConst = traverse(jet, jet["tree"][node_id][1], [], [])

    Im = abs(LConst - RConst) / (LConst + RConst)
    #         return LConst, RConst, abs(LConst - RConst)/(LConst+RConst)

    return Im


def traverse(jet, node_id, outers_list, Nconst):
    """
    Traverse function to get the number of constituents of a branch
    """
    if jet["tree"][node_id, 0] == -1:

        Nconst.append(1)
    #         print("Nconst = ", Nconst)
    #         outers_list.append(jet["content"][node_id])

    else:
        traverse(
            jet,
            jet["tree"][node_id, 0],
            outers_list,
            Nconst,
        )

        traverse(
            jet,
            jet["tree"][node_id, 1],
            outers_list,
            Nconst,
        )

    return len(Nconst)


"#########################################################"
""" PLOTTING """


def LogLHscatterPlot(truthDic, GreedyDic, BSODic, truth=True):
    """ Log LH scatter plot"""

    truthLogLH = truthDic["jetsListLogLH"]
    greedy_jetsLogLH = GreedyDic["jetsListLogLH"]
    BSO_jetsListLogLH = BSODic["jetsListLogLH"]

    fig2, (ax1) = plt.subplots(nrows=1, ncols=1)
    fig2.set_size_inches(7, 7)
    plt.tight_layout(pad=0.4, w_pad=5, h_pad=1.0)
    markersize = 1

    x = np.linspace(min(greedy_jetsLogLH), max(truthLogLH), 1000)

    if truth:

        ax1.scatter(truthLogLH, greedy_jetsLogLH, color="red", marker="X", s=markersize, label="Greedy")
        ax1.scatter(truthLogLH, BSO_jetsListLogLH, color='green', marker="o", s=markersize, label="Beam Search")

        ax1.plot(x, x, color="black", linestyle='--')
        ax1.set_xlabel(r"Truth Jet log likelihood", fontsize=15)
        ax1.set_ylabel(r"Clustered Jet log likelihood", fontsize=15)

    else:

        ax1.scatter(BSO_jetsListLogLH, greedy_jetsLogLH, color="red", marker="X", s=markersize, label="Greedy")
        ax1.plot(x, x, color="green", linestyle='--')

        ax1.set_xlabel(r"Beam Search jet log likelihood", fontsize=15)
        ax1.set_ylabel(r"Greedy jet log likelihood", fontsize=15)

    ax1.grid(which='both', axis='both', linestyle='--')
    plt.legend(loc='best', fontsize=15)
    plt.grid(which='both', axis='both', linestyle='--')

    plt.show()




# def variableHist(truthDic,
#              GreedyDic,
#              BSODic,
#              bins=50,
#              density=None,
#              normed=None,
#              variable = "deltaRoot",
#              name = "$\Delta_{root}$",
#              fixedJetP = False):
#     """ Delta root histogram """
#
#     fig2, (ax1) = plt.subplots(nrows=1, ncols=1)
#     fig2.set_size_inches(7, 7)
#
#     '''Same bins for all histograms'''
#     bins = np.histogram(np.hstack((truthDic[variable], BSODic[variable], GreedyDic[variable])), bins=bins)[
#         1]  # get the bin edges
#
#     plt.hist(truthDic[variable],
#              density=density,
#              bins=bins,
#              normed=normed,
#              histtype="step",
#              fill=False,
#              align='mid',
#              label="Truth",
#              color="black")
#
#     plt.hist(GreedyDic[variable],
#              density=density,
#              bins=bins,
#              normed=normed,
#              histtype="step",
#              fill=False,
#              align='mid',
#              label="Greedy",
#              color="r")
#
#     plt.hist(BSODic[variable],
#              density=density,
#              bins=bins,
#              normed=normed,
#              histtype="step",
#              fill=False,
#              align='mid',
#              label="Beam Search",
#              color="g")
#
#     if fixedJetP:
#         root_id = truthDic["jetsList"][0]["root_id"]
#         root_p = truthDic["jetsList"][0]["content"][root_id]
#
#         jetAngle = np.arctan2(root_p[0], root_p[1])
#         plt.axvline(jetAngle, color="black", linestyle='--')
#
#
#     plt.xlabel(r"%s "%name, fontsize=15)
#     plt.ylabel(" # of Jets", fontsize=15)
#     plt.legend(loc='best', fontsize=15)
#     plt.grid(which='both', axis='both', linestyle='--')
#
#     plt.show()


def variableHist(truthDic, variable1 = None,
             variable2 = None,
             variable3 = None,
             bins=50,
             density=None,
             normed=None,
             name = "$\Delta_{root}$",
             yaxisName = None,
             fixedJetP = False,
            labelLoc = None):
    """ Delta root histogram """

    fig2, (ax1) = plt.subplots(nrows=1, ncols=1)
    fig2.set_size_inches(7, 7)

    '''Same bins for all histograms'''
    bins = np.histogram(np.hstack((variable1, variable2, variable3)), bins=bins)[
        1]  # get the bin edges

    plt.hist(variable1,
             density=density,
             bins=bins,
             normed=normed,
             histtype="step",
             fill=False,
             align='mid',
             label="Truth",
             color="black")

    plt.hist(variable2,
             density=density,
             bins=bins,
             normed=normed,
             histtype="step",
             fill=False,
             align='mid',
             label="Greedy",
             color="r")

    plt.hist(variable3,
             density=density,
             bins=bins,
             normed=normed,
             histtype="step",
             fill=False,
             align='mid',
             label="Beam Search",
             color="g")

    if fixedJetP:
        root_id = truthDic["jetsList"][0]["root_id"]
        root_p = truthDic["jetsList"][0]["content"][root_id]

        jetAngle = np.arctan2(root_p[0], root_p[1])
        plt.axvline(jetAngle, color="black", linestyle='--')


    plt.xlabel(r"%s "%name, fontsize=15)
    plt.ylabel(" # of Jets", fontsize=15)
    if yaxisName:
        plt.ylabel(r"%s " % yaxisName, fontsize=15)

    plt.legend(loc='best', fontsize=15)
    if labelLoc:
        plt.legend(loc="%s"%labelLoc, fontsize=15)
    plt.grid(which='both', axis='both', linestyle='--')

    plt.show()






def PtscatterPlot(truthDic, GreedyDic, BSODic, dicString="SubjetPyMin", Greedy=False, BS=False, diff=False):
    dicString = dicString

    fig2, (ax1) = plt.subplots(nrows=1, ncols=1)
    fig2.set_size_inches(7, 7)
    plt.tight_layout(pad=0.4, w_pad=5, h_pad=1.0)
    markersize = 0.8

    x = np.linspace(min(truthDic[dicString]), max(truthDic[dicString]), 1000)

    if Greedy:

        if diff:
            ax1.scatter(truthDic["SubjetPyDiff"],
                        GreedyDic["SubjetPyDiff"],
                        color="Green",
                        marker="X",
                        s=markersize,
                        label="Greedy")

            x = np.linspace(min(truthDic["SubjetPyDiff"]), max(truthDic["SubjetPyDiff"]), 1000)

        else:
            ax1.scatter(truthDic[dicString],
                        GreedyDic[dicString],
                        color="Green",
                        marker="X",
                        s=markersize,
                        label="Greedy")

        ax1.plot(x, x, color="red", linestyle='--')
        ax1.set_xlabel(r"Truth Subjet Py", fontsize=15)
        ax1.set_ylabel(r"Greedy Subjet Py", fontsize=15)

    elif BS:

        if diff:
            ax1.scatter(truthDic["SubjetPyDiff"],
                        BSODic["SubjetPyDiff"],
                        color='blue',
                        marker="o",
                        s=markersize,
                        label="Beam Search")

            x = np.linspace(min(truthDic["SubjetPyDiff"]), max(truthDic["SubjetPyDiff"]), 1000)

        else:
            ax1.scatter(truthDic[dicString],
                        BSODic[dicString],
                        color='blue',
                        marker="o",
                        s=markersize,
                        label="Beam Search")

        ax1.plot(x, x, color="red", linestyle='--')
        ax1.set_xlabel(r"Truth Subjet Py", fontsize=15)
        ax1.set_ylabel(r"Beam Search Subjet Py", fontsize=15)

    else:

        ax1.scatter(BSODic[dicString],
                    GreedyDic[dicString],
                    color="Green",
                    marker="X",
                    s=markersize,
                    label="Greedy")

        x = np.linspace(min(BSODic[dicString]), max(BSODic[dicString]), 1000)

        ax1.plot(x, x, color="red", linestyle='--')
        ax1.set_xlabel(r"BS Subjet Py", fontsize=15)
        ax1.set_ylabel(r"Greedy Subjet Py", fontsize=15)

    plt.legend(loc='best', fontsize=15)
    plt.grid(which='both', axis='both', linestyle='--')

    plt.show()


def ConstPhiHist(truthDic, bins=50, density=False, fixedJetP=False):
    """ Subjets constituents angle. The origin is in the beam axiz (z direction)"""

    fig2, (ax1) = plt.subplots(nrows=1, ncols=1)
    fig2.set_size_inches(7, 7)

    plt.hist(truthDic["JetsConstPhi"],
             density=density,
             bins=bins,
             histtype="step",
             fill=False,
             align='mid',
             label="Truth",
             color="black")

    """ If samples have all the jets with the same momentum"""
    if fixedJetP:
        root_id = truthDic["jetsList"][0]["root_id"]
        root_p = truthDic["jetsList"][0]["content"][root_id]
        jetAngle = np.arctan2(root_p[0], root_p[1])
        plt.axvline(jetAngle, color="black", linestyle='--')

    plt.xlabel(" Jets constituents polar angle  ", fontsize=15)
    plt.ylabel(" # of jets constituents", fontsize=15)

    plt.legend(loc='best', fontsize=15)
    plt.grid(which='both', axis='both', linestyle='--')

    plt.show()













def dijLogLHscatter(variable = None,
             nameX = "Subjets $d_{ij}$",
            nameY ="Subjet splitting log likelihood",
                    title = None,
                    logLH = False,
                    dijOnly = False,
                    kt = False,
                    antikt = False,
                    jetdijs2 = None,
                    jetdijs3 = None,
                    LabelJetdijs = None,
                    LabelJetdijs2 = None,
                    LabelJetdijs3 = None,
                    ):

    jetdijs = variable

    fig2, (ax1) = plt.subplots(nrows=1, ncols=1)
    fig2.set_size_inches(5, 5)

    plt.tight_layout(pad=0.4, w_pad=5, h_pad=1.0)
    markersize = 1

    if logLH:
        ax1.scatter(jetdijs[0], jetdijs[3], color="green", marker="X", s=markersize, label="kT")
        ax1.scatter(jetdijs[0], jetdijs[2], color="blue", marker="X", s=markersize, label="CA")
        ax1.scatter(jetdijs[0], jetdijs[1], color="red", marker="X", s=markersize, label="Anti-kT")

    elif dijOnly:
        ax1.scatter(jetdijs[3], jetdijs[1], color="black", marker="X", s=markersize, label="")

    elif kt:
        ax1.scatter(jetdijs[0], jetdijs[3], color="green", marker="X", s=markersize, label=LabelJetdijs)
        ax1.scatter(jetdijs2[0], jetdijs2[3], color="blue", marker="X", s=markersize, label=LabelJetdijs2)
        ax1.scatter(jetdijs3[0], jetdijs3[3], color="red", marker="X", s=markersize, label=LabelJetdijs3)

    elif antikt:
        ax1.scatter(jetdijs[0], jetdijs[1], color="green", marker="X", s=markersize, label=LabelJetdijs)
        ax1.scatter(jetdijs2[0], jetdijs2[1], color="blue", marker="X", s=markersize, label=LabelJetdijs2)
        ax1.scatter(jetdijs3[0], jetdijs3[1], color="red", marker="X", s=markersize, label=LabelJetdijs3)

    ax1.set_xlabel(r"%s "%nameX, fontsize=15)
    ax1.set_ylabel(r"%s "%nameY, fontsize=15)

    ax1.grid(which='both', axis='both', linestyle='--')
    plt.legend(loc='best', fontsize=15)
    plt.grid(which='both', axis='both', linestyle='--')
    plt.title(r"%s"%title, fontsize=20)

    plt.show()


