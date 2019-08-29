import copy
import numpy as np
from scripts.greedyclusteringTJS import merge_TJS, get_loglh_TJS, initialize_jet, finalize_jet, get_dangling_leaves

"""
This is a brute force (B*N^3 time) implementation of beam search with beam size B
based on ToyJetsShower jets
"""


def get_dangling_leaveslist(jetlist):
    idleaveslist = []
    for jet in jetlist:
        idleaves = get_dangling_leaves(jet)
        idleaveslist.append(idleaves)
    nleaves = len(idleaves)
    return idleaveslist, nleaves


def get_top_pairs(jetlist, idleaveslist, beamsize, delta_min, rate):
    """
    Return (id1, id2, maxlh, idjet)
    where i and j are the id of the leaves giving maximum likelihood
    maxlh is their likelihood
    idjet is the jet the belong to
    """
    table = []
    for idjet, idleaves in enumerate(idleaveslist):
        for i, id1 in enumerate(idleaves):
            for j in range(i + 1, len(idleaves)):
                id2 = idleaves[j]
                table.append(
                    (
                        id1,
                        id2,
                        get_loglh_TJS(
                            jetlist[idjet]["content"][id1],
                            jetlist[idjet]["content"][id2],
                            delta_min,
                            rate,
                        ) +
                        sum(jetlist[idjet]["logLH"])
                        ,
                        idjet,
                    )
                )
    return sorted(table, key=lambda t: -t[2])[:beamsize]


def update_jetlist(jetlist, beamlist):
    newjetlist = []
    for triple in beamlist:
        jet = copy.deepcopy(jetlist[triple[3]])
        inew = len(jet["content"])
        jet["content"].append(
            merge_TJS(jet["content"][triple[0]], jet["content"][triple[1]])
        )
        jet["tree"].append([triple[0], triple[1]])
        jet["parent"].append(None)
        jet["parent"][triple[0]] = inew
        jet["parent"][triple[1]] = inew
        jet["logLH"].append(triple[2]-np.sum(jet["logLH"]))
        newjetlist.append(jet)
    return newjetlist


def build_beamsearch_tree_TJS(leaves, beamsize, delta_min, rate):
    """
    Given a set of leaves cluster them
    according to a beam search algorithm based on TJS generative model
    """
    jet = initialize_jet(leaves)
    jetlist = [jet]
    idleaves = get_dangling_leaves(jet)
    idleaveslist = [idleaves]
    beamlist = get_top_pairs(jetlist, idleaveslist, beamsize, delta_min, rate)
    jetlist = update_jetlist(jetlist, beamlist)
    idleaveslist, nleaves = get_dangling_leaveslist(jetlist)
    while nleaves > 1:
        beamlist = get_top_pairs(jetlist, idleaveslist, beamsize, delta_min, rate)
        jetlist = update_jetlist(jetlist, beamlist)
        idleaveslist, nleaves = get_dangling_leaveslist(jetlist)
    return finalize_jet(jetlist[0])
