import numpy as np
import logging
import pickle
import time
import importlib
import copy

from scripts import reclusterTree
from scripts import linkageList
from scripts import heatClustermap
from scripts import Tree1D
from scripts import likelihood
from scripts import beamsearchTJS
from scripts import N2Greedy
from scripts import beamSearch as bs
from scripts import beamSearchOptimal as BSO
from scripts.utils import get_logger

logger = get_logger(level=logging.INFO)



data_dir="data/"

def appendTruthJets(start, end, Njets):
    """ Load truth trees and create logLH lists """

    dic = {}

    Total_jetsList = []
    Total_jetsListLogLH = []
    avg_logLH = []
    # Nconst = []

    for i in range(start, end):
        with open(data_dir+"truth/tree_" + str(Njets) + "_truth_" + str(i) + ".pkl", "rb") as fd:
            jetsList = pickle.load(fd, encoding='latin-1')

        # """Number of jet constituents"""
        # Nconst.append([len(jet["leaves"]) for jet in jetsList])

        """Fill jet dictionaries with log likelihood of truth jet"""
        [likelihood.enrich_jet_logLH(jet, dij=True) for jet in jetsList]

        enrichTruthLogLH = [np.sum(jet["logLH"]) for jet in jetsList]

        Total_jetsList.append(jetsList)
        Total_jetsListLogLH.append(enrichTruthLogLH)
        avg_logLH.append(np.average(enrichTruthLogLH))

    """ Standard deviation for the average log LH for the N runs"""
    sigma = np.std(avg_logLH)

    """ Statistical error for the mean log LH for the  total number of jets as err = sqrt(s)/ sqrt(N), where  s is the sample variance"""
    flatTotal_jetsListLogLH = np.asarray(Total_jetsListLogLH).flatten()
    statSigma = np.std(flatTotal_jetsListLogLH) / np.sqrt(len(flatTotal_jetsListLogLH))

    dic["jetsList"] = Total_jetsList
    # dic["NconstList"] = np.asarray(Nconst)
    dic["jetsListLogLH"] = flatTotal_jetsListLogLH
    dic["avgLogLH"] = np.asarray(avg_logLH)
    dic["sigma"] = sigma
    dic["statSigma"] = statSigma

    return dic





def appendGreedyJets(start, end, Njets):
    """ Load greedy trees and logLH lists """

    startTime = time.time()

    dic = {}

    Total_jetsList = []
    Total_jetsListLogLH = []
    avg_logLH = []
    for i in range(start, end):
        with open(data_dir+"GreedyJets/Greedy_" + str(Njets) + "Mw_" + str(i) + ".pkl", "rb") as fd:
            jetsList, _ = pickle.load(fd, encoding='latin-1')

        """ Fill deltas list (needed to fill the jet log LH)"""
        [traverseTree(jet) for jet in jetsList]

        [likelihood.fill_jet_info(jet, parent_id=None) for jet in jetsList]

        """Fill jet dictionaries with log likelihood of truth jet"""
        [likelihood.enrich_jet_logLH(jet, dij=True) for jet in jetsList]

        jetsListLogLH = [np.sum(jet["logLH"]) for jet in jetsList]

        Total_jetsList.append(jetsList)
        Total_jetsListLogLH.append(jetsListLogLH)
        avg_logLH.append(np.average(jetsListLogLH))


    """ Standard deviation for the average log LH for the N runs"""
    sigma = np.std(avg_logLH)

    """ Statistical error for the mean log LH for the  total number of jets as err = sqrt(s)/ sqrt(N), where sigma s the sample variance"""
    flatTotal_jetsListLogLH = np.asarray(Total_jetsListLogLH).flatten()
    statSigma = np.std(flatTotal_jetsListLogLH) / np.sqrt(len(flatTotal_jetsListLogLH))

    dic["jetsList"] = Total_jetsList
    dic["jetsListLogLH"] = flatTotal_jetsListLogLH
    dic["avgLogLH"] = np.asarray(avg_logLH)
    dic["sigma"] = sigma
    dic["statSigma"] = statSigma

    logger.info(f" TOTAL TIME = {time.time() - startTime}")

    return dic





def appendBSO_Scan(start, end, Njets):
    """ Load beam search trees and logLH lists """

    startTime = time.time()

    dic = {}

    Total_jetsList = []
    Total_jetsListLogLH = []
    avg_logLH = []
    for i in range(start, end):
        with open(data_dir+"BeamSearchJets/BSO_" + str(Njets) + "Mw_" + str(i) + ".pkl", "rb") as fd:
            jetsList, _ = pickle.load(fd, encoding='latin-1')

        """ Fill deltas list (needed to fill the jet log LH)"""
        [traverseTree(jet) for jet in jetsList]

        [likelihood.fill_jet_info(jet, parent_id=None) for jet in jetsList]

        """Fill jet dictionaries with log likelihood of truth jet"""
        [likelihood.enrich_jet_logLH(jet, dij=True) for jet in jetsList]

        jetsListLogLH = [np.sum(jet["logLH"]) for jet in jetsList]


        Total_jetsList.append(jetsList)
        Total_jetsListLogLH.append(jetsListLogLH)
        avg_logLH.append(np.average(jetsListLogLH))


    """ Standard deviation for the average log LH for the N runs"""
    sigma = np.std(avg_logLH)

    """ Statistical error for the mean log LH for the  total number of jets as err = sqrt(s)/ sqrt(N), where sigma s the sample variance"""
    flatTotal_jetsListLogLH = np.asarray(Total_jetsListLogLH).flatten()
    statSigma = np.std(flatTotal_jetsListLogLH) / np.sqrt(len(flatTotal_jetsListLogLH))

    dic["jetsList"] = Total_jetsList
    dic["jetsListLogLH"] = flatTotal_jetsListLogLH
    dic["avgLogLH"] = np.asarray(avg_logLH)
    dic["sigma"] = sigma
    dic["statSigma"] = statSigma

    logger.info(f" TOTAL TIME = {time.time() - startTime}")

    return dic





def traverseTree(jet):

    """ Traverse jet to get ancestors list  and content list starting from the root node"""
    tree, \
    content, \
    node_id, \
    tree_ancestors = N2Greedy._traverse(
        jet["root_id"],
        jet["content"],
        jetTree=jet["tree"],
        Nleaves=jet["Nconst"],
    )

    jet["root_id"] = 0
    jet["node_id"] = node_id
    jet["tree"] = np.asarray(tree).reshape(-1, 2)
    jet["content"] = np.asarray(content).reshape(-1, 2)
    jet["tree_ancestors"] = tree_ancestors




""" RUN GREEDY AND BEAM SEARCH ALGORITHMS """

def fill_GreedyList(input_jets, Nbest=1, k1=0, k2=2):
    """ Run the greedy algorithm over a list of sets of input jets.
        Args: input jets
        returns: clustered jets
                     jets logLH
    """

    input_dir = "data/truth/"

    with open(input_dir + str(input_jets) + '.pkl', "rb") as fd:
        truth_jets = pickle.load(fd, encoding='latin-1')[k1:k2]

    startTime = time.time()

    greedyJets = [N2Greedy.recluster(
        truth_jet,
        delta_min=truth_jet["pt_cut"],
        lam=float(truth_jet["Lambda"]),
        visualize=False,
    ) for truth_jet in truth_jets]

    print("TOTAL TIME = ", time.time() - startTime)

    greedyJetsLogLH = [sum(jet["logLH"]) for jet in greedyJets]

    return greedyJets, greedyJetsLogLH


def fill_BSList(input_jets, Nbest=1, k1=0, k2=2):
    """ Run the Beam search algorithm (algorithm where when the logLH of 2 or more trees is the same, we only keep one of them) over a list  of sets of input jets.
        Args: input jets
        returns: clustered jets
                     jets logLH
    """

    input_dir = "data/truth/"

    with open(input_dir + str(input_jets) + '.pkl', "rb") as fd:
        truth_jets = pickle.load(fd, encoding='latin-1')[k1:k2]

    startTime = time.time()
    BSO_jetsList = []

    a = time.time()

    for i, truth_jet in enumerate(truth_jets):

        if i % 50 == 0:
            print(" # of reclustered jets = ", i, "; Partial time = ", time.time() - a)
            #                 print("PARTIAL TIME = ",time.time() -a)
            a = time.time()

        N = len(truth_jet["leaves"])

        BSO_jetsList.append(BSO.recluster(
            truth_jet,
            beamSize=min(3 * N, np.asarray(N * (N - 1) / 2).astype(int)),
            delta_min=truth_jet["pt_cut"],
            lam=float(truth_jet["Lambda"]),
            N_best=Nbest,
        )[0]
                            )

    print("TOTAL TIME = ", time.time() - startTime)

    BSO_jetsListLogLH = [sum(jet["logLH"]) for jet in BSO_jetsList]

    return BSO_jetsList, BSO_jetsListLogLH