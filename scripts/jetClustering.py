import numpy as np
import logging
import pickle
import time
import importlib
import copy
import argparse
import os

from scripts import reclusterTree
from scripts import linkageList
from scripts import heatClustermap
from scripts import likelihood
from scripts import N2Greedy
from scripts import beamSearchOptimal as BSO
from scripts.utils import get_logger

logger = get_logger(level=logging.INFO)



# data_dir="/scratch/sm4511/TreeAlgorithms/data/"
data_dir="/Users/sebastianmacaluso/Documents/PrinceData/"





"""####################################"""

def appendJets(start, end, Njets, truth = False, BS = False, Greedy = False):
    """ Load truth trees and create logLH lists """

    startTime = time.time()

    dic = {}

    Total_jetsList = []
    Total_jetsListLogLH = []
    avg_logLH = []
    # Nconst = []

    TruthFilename = "Truth/tree_" + str(Njets) + "_truth_"
    BSFilename = "GreedyJets/Greedy_" + str(Njets) + "Mw_"
    greedyFilename = "BeamSearchJets/BSO_" + str(Njets) + "Mw_"


    for i in range(start, end):
        if  ( os.path.isfile(TruthFilename+ str(i) + ".pkl")
            and os.path.isfile(BSFilename+ str(i) + ".pkl")
            and os.path.isfile(greedyFilename+ str(i) + ".pkl")):

            with open(data_dir+"Truth/tree_" + str(Njets) + "_truth_" + str(i) + ".pkl", "rb") as fd:
                jetsList = pickle.load(fd, encoding='latin-1')

            # """Number of jet constituents"""
            # Nconst.append([len(jet["leaves"]) for jet in jetsList])

            # """Fill jet dictionaries with log likelihood of truth jet"""
            # [likelihood.enrich_jet_logLH(jet, dij=True) for jet in jetsList]

            enrichTruthLogLH = [np.sum(jet["logLH"]) for jet in jetsList]

            Total_jetsList.append(jetsList)
            Total_jetsListLogLH.append(enrichTruthLogLH)

            if (i+1)%20==0:
                avg_logLH.append(np.average(np.asarray(Total_jetsListLogLH[i-19:i+1]).flatten()))

    # print("lenght avg_logLH = ", len(avg_logLH))

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

    logger.info(f" TOTAL TIME = {time.time() - startTime}")

    return dic

""" ################################### """
def appendTruthJets(start, end, Njets):
    """ Load truth trees and create logLH lists """

    startTime = time.time()

    dic = {}

    Total_jetsList = []
    Total_jetsListLogLH = []
    avg_logLH = []
    # Nconst = []

    for i in range(start, end):
        with open(data_dir+"Truth/tree_" + str(Njets) + "_truth_" + str(i) + ".pkl", "rb") as fd:
            jetsList = pickle.load(fd, encoding='latin-1')

        # """Number of jet constituents"""
        # Nconst.append([len(jet["leaves"]) for jet in jetsList])

        # """Fill jet dictionaries with log likelihood of truth jet"""
        # [likelihood.enrich_jet_logLH(jet, dij=True) for jet in jetsList]

        enrichTruthLogLH = [np.sum(jet["logLH"]) for jet in jetsList]

        Total_jetsList.append(jetsList)
        Total_jetsListLogLH.append(enrichTruthLogLH)

        if (i+1)%20==0:
            avg_logLH.append(np.average(np.asarray(Total_jetsListLogLH[i-19:i+1]).flatten()))

    # print("lenght avg_logLH = ", len(avg_logLH))

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

    logger.info(f" TOTAL TIME = {time.time() - startTime}")

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
            jetsList, jetsListLogLH = pickle.load(fd, encoding='latin-1')

        # """ Fill deltas list (needed to fill the jet log LH)"""
        # [traverseTree(jet) for jet in jetsList]

        # [likelihood.fill_jet_info(jet, parent_id=None) for jet in jetsList]
        #
        # """Fill jet dictionaries with log likelihood of truth jet"""
        # [likelihood.enrich_jet_logLH(jet, dij=True) for jet in jetsList]

        # jetsListLogLH = [np.sum(jet["logLH"]) for jet in jetsList]

        Total_jetsList.append(jetsList)
        Total_jetsListLogLH.append(jetsListLogLH)

        if (i+1)%20==0:
            avg_logLH.append(np.average(np.asarray(Total_jetsListLogLH[i-19:i+1]).flatten()))

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
            jetsList, jetsListLogLH = pickle.load(fd, encoding='latin-1')

        # """ Fill deltas list (needed to fill the jet log LH)"""
        # [traverseTree(jet) for jet in jetsList]

        # [likelihood.fill_jet_info(jet, parent_id=None) for jet in jetsList]
        #
        # """Fill jet dictionaries with log likelihood of truth jet"""
        # [likelihood.enrich_jet_logLH(jet, dij=True) for jet in jetsList]

        # jetsListLogLH = [np.sum(jet["logLH"]) for jet in jetsList]


        Total_jetsList.append(jetsList)
        Total_jetsListLogLH.append(jetsListLogLH)

        if (i+1)%20==0:
            avg_logLH.append(np.average(np.asarray(Total_jetsListLogLH[i-19:i+1]).flatten()))

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





# def traverseTree(jet):
#
#     """ Traverse jet to get ancestors list  and content list starting from the root node"""
#     tree, \
#     content, \
#     node_id, \
#     tree_ancestors = N2Greedy._traverse(
#         jet["root_id"],
#         jet["content"],
#         jetTree=jet["tree"],
#         Nleaves=jet["Nconst"],
#     )
#
#     jet["root_id"] = 0
#     jet["node_id"] = node_id
#     jet["tree"] = np.asarray(tree).reshape(-1, 2)
#     jet["content"] = np.asarray(content).reshape(-1, 2)
#     jet["tree_ancestors"] = tree_ancestors




""" RUN GREEDY AND BEAM SEARCH ALGORITHMS """

def fill_GreedyList(input_jets, Nbest=1, k1=0, k2=2):
    """ Run the greedy algorithm over a list of sets of input jets.
        Args: input jets
        returns: clustered jets
                     jets logLH
    """



    with open(args.data_dir + str(input_jets) + '.pkl', "rb") as fd:
        truth_jets = pickle.load(fd, encoding='latin-1')[k1:k2]

    startTime = time.time()

    # for k,truth_jet in enumerate(truth_jets):
    #     print("k = ",k)
    #     if k==27:
    #         print("M_Hard = ",truth_jet["M_Hard"])
    #         # print("Nconst = ", truth_jet["Nconst"] )
    #     N2Greedy.recluster(
    #         truth_jet,
    #         delta_min=truth_jet["pt_cut"],
    #         lam=float(truth_jet["Lambda"]),
    #         visualize=True,
    #     )


    greedyJets = [N2Greedy.recluster(
        truth_jet,
        delta_min=truth_jet["pt_cut"],
        lam=float(truth_jet["Lambda"]),
        visualize = True,
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



    with open(args.data_dir + str(input_jets) + '.pkl', "rb") as fd:
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
            visualize = True,
        )[0]
                            )

    print("TOTAL TIME = ", time.time() - startTime)

    BSO_jetsListLogLH = [sum(jet["logLH"]) for jet in BSO_jetsList]

    return BSO_jetsList, BSO_jetsListLogLH



def fill_ktAlgos(input_jets, k1=0, k2=2, alpha = None):
    """ Run the generalized kt algorithm over a list of sets of input jets.
        Args: input jets
        returns: clustered jets
                     jets logLH
    """



    with open(args.data_dir + str(input_jets) + '.pkl', "rb") as fd:
        truth_jets = pickle.load(fd, encoding='latin-1')[k1:k2]

    startTime = time.time()

    generalizedKtjets = [reclusterTree.recluster(truth_jet, alpha=alpha, save=False)
                         for truth_jet in truth_jets]

    print("TOTAL TIME = ", time.time() - startTime)

    return generalizedKtjets


if __name__ == "__main__":

    # def runGreedy_Scan(start, end, Njets):
    #     """ Run greedy algorithm"""
    #
    #     for i in range(start, end):
    #         jetsList, jetsListLogLH = fill_GreedyList("tree_" + str(Njets) + "_truth_" + str(i), k1=0,
    #                                                   k2=Njets)
    #
    #         with open(data_dir + "GreedyJets/Greedy_" + str(Njets) + "Mw_" + str(i) + ".pkl", "wb") as f:
    #             pickle.dump((jetsList, jetsListLogLH), f)
    #
    #
    # def runBSO_Scan(start, end, Njets):
    #     """ Run beam search algorithm"""
    #
    #     for i in range(start, end):
    #         BSO_jetsList, BSO_jetsListLogLH = fill_BSList("tree_" + str(Njets) + "_truth_" + str(i), k1=0,
    #                                                       k2=Njets)
    #
    #         with open(data_dir + "BeamSearchJets/BSO_" + str(Njets) + "Mw_" + str(i) + ".pkl", "wb") as f:
    #             pickle.dump((BSO_jetsList, BSO_jetsListLogLH), f)

    def runGreedy_Scan(i, Njets):
        """ Run greedy algorithm"""

        jetsList, jetsListLogLH = fill_GreedyList("tree_" + str(Njets) + "_truth_" + str(i), k1=0,
                                                  k2=Njets)

        output_dir = args.output_dir+"GreedyJets/"
        os.system('mkdir -p ' + output_dir)

        with open(output_dir+"Greedy_" + str(Njets) + "_" + str(i) + ".pkl", "wb") as f:
            pickle.dump((jetsList, jetsListLogLH), f)


    def runBSO_Scan(i, Njets):
        """ Run beam search algorithm"""

        BSO_jetsList, BSO_jetsListLogLH = fill_BSList("tree_" + str(Njets) + "_truth_" + str(i), k1=0,
                                                      k2=Njets)

        output_dir = args.output_dir+"BeamSearchJets/"
        os.system('mkdir -p ' + output_dir)

        with open(output_dir+"BSO_" + str(Njets) + "_" + str(i) + ".pkl", "wb") as f:
            pickle.dump((BSO_jetsList, BSO_jetsListLogLH), f)



    def runKtAntiKtCA_Scan(i, Njets, alpha=None):
        """ Run beam search algorithm"""
        generalizedKtjets = fill_ktAlgos("tree_" + str(Njets) + "_truth_" + str(i),
                                         k1=0,
                                         k2=Njets,
                                         alpha=alpha)

        if alpha == 1:
            name = "Kt"
        elif alpha == -1:
            name = "Antikt"
        elif alpha == 0:
            name = "CA"
        else:
            raise ValueError(f"Please pick a valid value for alpha (e.g. -1,0,1)")

        output_dir = args.output_dir+"/"+name+"Jets/"
        os.system('mkdir -p ' + output_dir)

        with open(output_dir + name+"_" + str(Njets) + "_" + str(i) + ".pkl", "wb") as f:
            pickle.dump(generalizedKtjets, f)



    parser = argparse.ArgumentParser(description="Run Greedy and Beam Search algorithms")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increase output verbosity"
    )

    parser.add_argument(
        "--jetType", type=str, required=True, help="Input jet type, e.g. 'QCDjets' or 'Wjets' "
    )

    parser.add_argument(
        "--greedyScan", type=str, default="False", help="Flag to run greedy clustering"
    )
    parser.add_argument(
        "--BSScan", type=str, default="False", help="Flag to run beam seach clustering"
    )

    parser.add_argument(
        "--KtAntiktCAscan", type=str, default="False", help="Flag to run generalized kt clustering"
    )

    parser.add_argument(
        "--id", type=str, default=0, help="dataset id"
    )

    parser.add_argument(
        "--N_jets", type=str, default=2, help="# of jets in each dataset"
    )

    args = parser.parse_args()

    parser.add_argument(
        "--data_dir", type=str, default="/scratch/sm4511/TreeAlgorithms/data/"+args.jetType+"/Truth/", help="Data dir"
    )

    parser.add_argument(
        "--output_dir", type=str, default="/scratch/sm4511/TreeAlgorithms/data/"+args.jetType+"/", help="Output dir"
    )



    logger = get_logger(level=logging.INFO)

    args = parser.parse_args()

    data_dir = args.data_dir

    # To test:
    # Nstart = 0
    # Nend = 4
    # N_jets = 2

    # Full dataset
    # Nstart = 10
    # Nend = 30
    # N_jets = 500

    """We ran a scan for 30 sets of 500 jets each."""
    if args.greedyScan == "True":
        runGreedy_Scan(int(args.id), int(args.N_jets))
        # runGreedy_Scan(Nstart, Nend, N_jets)



    """We ran a scan for 10 sets of 500 jets each. (Below as an example there is a scan for 4 sets of 2 jets each)"""
    if args.BSScan == "True":
        runBSO_Scan(int(args.id), int(args.N_jets))
        # runBSO_Scan(Nstart, Nend, N_jets)


    """We ran a scan for 10 sets of 500 jets each. (Below as an example there is a scan for 4 sets of 2 jets each)"""
    if args.KtAntiktCAscan == "True":
        for alphaValue in [-1,0,1]:
            runKtAntiKtCA_Scan(int(args.id), int(args.N_jets), alpha = alphaValue)
