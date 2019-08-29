import sys
import numpy as np
import logging
import pickle
import itertools
import copy
import heapq
import time
import bisect

from scripts import likelihood
from scripts import N2Greedy

from scripts.utils import get_logger

logger = get_logger(level=logging.INFO)


"""
Beam search algorithm:
1) Get all the possible pairings log likelihood for the leaves of the tree and sort them. O(N^2 logN)
2) for each level:
	for each beam push the max log LH pairs to a priority queue of size b (b is the beam size) O(b^2 log b)
	update each tree => O(b N^2 log N)
	
Total: O( b^2 N log b + b N^3 log N). Typically b > N, but for b log < N^2 log N,  we get O(b N^3 log N)

"""


class latentPath(object):
	"""
	Class that stores the jet dictionary information for each latent path included in the beam search algorithm:

		- levelContent: nodes list after deleting the constituents that are merged and adding the new pseudojet in each level.
		  So this should only have the root of the tree at the end.

		- levelDeltas: list with the delta value (for the splitting of a parent node) of each node. Zero if a leaf.

		- logLH: list with the log likelihood of each pairing.

		- jetContent: array with the momentum of all the nodes of the jet tree (both leaves and inners).

		- tree_dic: dictionary that has the node id of a parent as a key and a list with the id of the 2 children as the values

		- idx: array that stores the node id
		(the node id determines the location of the momentum vector of a pseudojet in the jet_content array)
		of the pseudojets that are in the current content_level array. It has the same elements as the content_level (they get updated
		level by level).

		- N_leaves_list: List that given a node idx, stores for that idx, the number of leaves for the branch below that node. It is initialized only with the tree leaves

		- linkage_list: linkage list to build heat clustermap visualizations.
	"""

	def __init__(
			self,
			path_sortPairs = None,
			path_levelContent = None,
			path_levelDeltas = None,
			path_logLH = None,
			path_jetTree = None,
			path_jetContent =None,
			path_idx = None,
			path_N_leaves_list = None,
			path_linkage_list = None,

	):

		self.sortPairs = path_sortPairs
		self.levelContent = path_levelContent
		self.levelDeltas = path_levelDeltas
		self.logLH = path_logLH
		self.jetTree = path_jetTree
		self.jetContent =path_jetContent
		self.idx = path_idx
		self.N_leaves_list = path_N_leaves_list
		self.linkage_list = path_linkage_list






def recluster(
		jet_dic,
		save = False,
		delta_min = None,
		lam = None,
		beamSize = None,
		N_best = None,
		visualize = False,
):
	"""
	Get the leaves of an  input jet,
	recluster them following the beam search algorithm determined by the beam size,
	create the new tree for the chosen algorithm,
	make a jet dictionary list with all with the jets that give the best log likelihood up to N_best and save it.

	Create a dictionary with all the jet tree info
		- jet["root_id"]: root node id of the tree
		- jet["content"]: list with the tree nodes (particles) momentum vectors. For the ToyJetsShower we consider a 2D model,
		so we have (py,pz), with pz the direction of the beam axis
		- jet["tree"]: list with the tree structure. Each entry contains a list with the [left,right] children of a node.
		If [-1,-1] then the node is a leaf.

	New features added to the tree:
		- jet["tree_ancestors"]: List with one entry for each leaf of the tree, where each entry lists all the ancestor node ids
		when traversing the tree from the root to the leaf node.
		- jet["linkage_list"]: linkage list to build heat clustermap visualizations.
		- jet["Nconst"]: Number of leaves of the tree.
		- jet["algorithm"]: Algorithm to generate the tree structure, e.g. truth, kt, antikt, CA.

	Args:
		- jet_dic: any jet dictionary with the clustering history.

		- beamSize: beam size for the beam search algorithm, i.e. it determines the number of trees latent path run in parallel and kept in memory.

		- delta_min: pT cut scale for the showering process to stop.

		- lam: decaying rate value for the exponential distribution.

		- N_best = Number of jets that give the best log likelihood to be kept in memory

		- save: if true, save the reclustered jet dictionary list

	Returns:
		- jetsList: List of jet dictionaries
	"""
	startTime = time.time()

	""" Get jet constituents list (tree leaves) """
	jet_const = N2Greedy.getConstituents(
		jet_dic,
		jet_dic["root_id"],
		[],
	)


	reclustStartTime = time.time()

	bestLogLH_paths, \
	root_node = beamSearch(
		jet_const,
		delta_min = delta_min,
		lam = lam,
		beamSize = beamSize,
	)


	""" Create jet dictionary with tree features for the best N_best trees (max likelihood criteria)"""

	jetsList = []

	for path in bestLogLH_paths[-N_best::]:

		jet = {}
		jet["root_id"] = root_node
		jet["tree"] = np.asarray(path.jetTree).reshape(-1, 2)
		jet["content"] = np.asarray(path.jetContent).reshape(-1, 2)
		jet["linkage_list"]=path.linkage_list
		jet["Nconst"]=len(jet_const)
		jet["algorithm"]= "beamSearch"
		jet["pt_cut"] = delta_min
		jet["Lambda"] = lam
		jet["logLH"] = np.asarray(path.logLH)



		""" Extra features needed for visualizations """
		if visualize:

			tree, \
			content, \
			node_id, \
			tree_ancestors = N2Greedy._traverse(
				root_node,
				path.jetContent,
				jetTree=path.jetTree,
				Nleaves=len(jet_const),
			)

			jet["node_id"] = node_id
			jet["tree"] = np.asarray(tree).reshape(-1, 2)
			jet["content"] = np.asarray(jetContent).reshape(-1, 2)
			jet["tree_ancestors"] = tree_ancestors


		jetsList.append(jet)


	logger.debug(f" Recluster and build tree algorithm total time = {time.time() - reclustStartTime}")

	# jetsList = jetsList[::-1]


	""" Save reclustered tree """
	if save:
		out_dir = "data/"
		algo = str(jet_dic["name"]) + '_' + str(alpha)
		out_filename = out_dir + str(algo) + '.pkl'

		logger.info(f"Output jet filename = {out_filename}")

		with open(out_filename, "wb") as f:
			pickle.dump(jetsList, f, protocol=2)


	logger.debug(f" Algorithm total time = {time.time() - startTime}")


	return jetsList







def beamSearch(
		levelContent,
		delta_min = None,
		lam = None,
		beamSize = None,
):
	"""
	Runs the level_SortedLogLH_beamPairs function level by level starting from the list of constituents (leaves) until we reach the root of the tree.
	Note: We refer to both leaves and inner nodes as pseudojets.

	Args:
		- constituents: jet constituents (i.e. the leaves of the tree)
		- beamSize: beam size for the beam search algorithm, i.e. it determines the number of trees latent path run in parallel and kept in memory
		- delta_min: pT cut scale for the showering process to stop.
		- lam: decaying rate value for the exponential distribution.

	Returns:

		- root_node: root node idx.
		- predecessors: Stores the jet dictionary information for each latent path included in the beam search algorithm. Each entry is an object defined by the latentPath class.

	"""

	Nconst = len(levelContent)

	jetTree = [[-1,-1]]*Nconst

	idx = [i for i in range(Nconst)]

	jetContent = copy.deepcopy(levelContent)

	root_node = 2 * Nconst - 2

	N_leaves_list = [1.] * Nconst

	linkage_list = []

	logLH = []

	levelDeltas = [0.] * Nconst


	""" Calculate the sorted list of all pairs based on max log likelihood (logLH) for each leaf of the tree. O(2 N^2 logN)"""
	sortPairs =  sortedPairs(
			levelContent,
			levelDeltas,
			Nconst = Nconst,
			delta_min = delta_min,
			lam = lam,
	)


	""" Initialize list that keeps track of best beam size latent paths jet trees """
	predecessors = [ ]

	path = latentPath(
		path_sortPairs = sortPairs,
		path_levelContent = levelContent,
		path_levelDeltas = levelDeltas,
		path_logLH=logLH,
		path_jetTree = jetTree,
		path_jetContent = jetContent,
		path_idx = idx,
		path_N_leaves_list=N_leaves_list,
		path_linkage_list=linkage_list,
	)

	predecessors.append(path)


	""" Loop over levels and cluster best node pairing for beam size best latent paths in each level"""
	for level in range(Nconst - 1):

		logger.debug(f"===============================================")
		logger.debug(f" LEVEL = {level}")
		logger.debug(f" LENGTH PREDECESSORS = {len(predecessors)}")


		""" Initialize priority queue to keep only beam size best latent paths (max log LH) at each level:
		 (sumLogLH, beamIdx, MaxPairIdx, pairlogLH) """
		pQueue = [(- np.inf, -9999, [-9999, -9999], -9999)] * beamSize
		heapq.heapify(pQueue)


		for j in range(len(predecessors)):

			pQueue = level_SortedLogLH_beamPairs (
				predecessor = predecessors[j],
				beamSize = beamSize,
				levelQueue = pQueue,
				beamIdx = j,
			)

		logger.debug(f" Best latent paths in priority queue  = {np.asarray(pQueue)}")


		""" Update latent paths (remove clustered nodes and add new one) and store them in predecessors """
		predecessors = updateLevelPaths(
			best_LevelPaths = pQueue,
			prevPredecessors = predecessors,
			Nconst = Nconst,
			Nparent = Nconst + level,
			delta_min=delta_min,
			lam=lam,
		)


	return predecessors, root_node






def sortedPairs(
    levelContent,
	levelDeltas,
    Nconst=None,
	delta_min = None,
	lam = None
):
	"""
	-For each leaf i of the tree, calculate its nearest neighbor (NN) j and the log likelihood for that pairing. This is O(N^2)
	Format: NNpairs = [ (logLH,[leaf i,leaf j]) for i in leaves]

	Calculate the log likelihood between all possible pairings of the leaves at a certain level and get the maximum.
	-Update the constituents list by deleting the constituents that are merged and adding the new pseudojet
	(We refer to both leaves and inner nodes as pseudojets.)

	Args:
	    - levelContent: array with the constituents momentum list for the current level (i.e. after deleting the constituents that are merged and
	      adding the new node from merging them in all previous levels)
	    - levelDeltas: array with the delta values (for the splitting of a node in the Toy Jets Shower Model) for the current level.
	    - Nconst: Number of leaves
	    - delta_min: pT cut scale for the showering process to stop.
		- lam: decaying rate value for the exponential distribution.

	Returns:
		- pairs

	"""

	pairs =  [
				           (
					           likelihood.Basic_split_logLH(
						           levelContent[k],
						           levelDeltas[k],
						           levelContent[k - j],
						           levelDeltas[k - j],
						           delta_min,
						           lam,
					           ),
					           [k, k - j]
				           )
		           for k in range(1, Nconst, 1)
		           for j in range(k, 0, -1)
	           ]

	# pairs = sorted(pairs, key = lambda x: x[0])

	dtype = [('logLH', float), ('pair', object)]
	b = np.array(pairs, dtype=dtype)
	pairs = np.sort(b, order='logLH')

	# logger.info(f" best pairs = {pairs[-10::]}")

	return pairs







def level_SortedLogLH_beamPairs(
	predecessor = None,
	beamSize = None,
	levelQueue = None,
	beamIdx = None,
):
	"""
	-Calculate all splitting log likelihood between all possible pair of constituents at a certain level and sort them.
	levelQueue = (total logLH, beamIdx, [node i,node j], level logLH )

	Args:
	    - in_levelContent: array with the constituents momentum list for the current level (i.e. deleting the constituents that are merged and
	      adding the new pseudojet from merging them)
	    - in_levelDeltas: list with the delta value (for the splitting of a parent node) of each node. Zero if a leaf.
		- beamSize: beam size for the beam search algorithm, i.e. it determines the number of trees latent path run in parallel and kept in memory
		- delta_min: pT cut scale for the showering process to stop.
		- lam: decaying rate value for the exponential distribution.

	Returns:
	    - maxPairs: top beamSize pairings at current level (logLH, indexes in pairs list) the give the greatest log likellihood

	"""

	beamPairs = predecessor.sortPairs[-beamSize::]
	SumLogLH = np.sum(predecessor.logLH)


	[heapq.heappushpop(levelQueue ,
			                        (SumLogLH + entry[0],
			                         beamIdx,
			                         entry[1],
			                         entry[0])
			                        ) for entry in beamPairs]


	return levelQueue







def updateLevelPaths(
		best_LevelPaths = None,
		prevPredecessors = None,
		Nconst= None,
		Nparent = None,
		delta_min=None,
		lam=None,
):
	"""
	Update the jet dictionary information by deleting the constituents that are merged and adding the new pseudojets
    (We refer to both leaves and inner nodes as pseudojets.)

    Args:
		best_LevelPaths: List with the best latent paths (beamIdx, total Log Likelihood, Max Pair Idx and last pairing log likelihood)

		prevPredecessors: predecessors list with the the jet dictionary information for the current best latent paths.

		Nconst: Number of leaves

		Nparent: parent idx for current pairing

	returns:
		-updatedPredecessors: updated predecessors list after adding current pairing.

	"""

	updatedPredecessors = []

	""" Dequeue each item (in increasing order of logLH )"""
	for k in range(len(best_LevelPaths)):

		logger.debug(f" ------------------------------- ")
		logger.debug(f" beam number = {k}")

		(SumLogLH, beamIdx, maxPairIdx, maxPairLogLH) = heapq.heappop(best_LevelPaths)


		logger.debug(f" (SumLogLH, maxPairLogLH) = {SumLogLH, maxPairLogLH}")
		logger.debug(f" prev logLH = {prevPredecessors[beamIdx].logLH}")
		logger.debug(f" ")

		beamIdx = int(beamIdx)


		sortPairs = copy.copy(prevPredecessors[beamIdx].sortPairs)
		levelContent = copy.copy(prevPredecessors[beamIdx].levelContent)
		levelDeltas = copy.copy(prevPredecessors[beamIdx].levelDeltas)
		logLH = copy.copy(prevPredecessors[beamIdx].logLH)
		N_leaves_list = copy.copy(prevPredecessors[beamIdx].N_leaves_list)
		linkage_list = copy.copy(prevPredecessors[beamIdx].linkage_list)
		jetTree = copy.copy(prevPredecessors[beamIdx].jetTree)
		jetContent = copy.copy(prevPredecessors[beamIdx].jetContent)
		idx = copy.copy(prevPredecessors[beamIdx].idx)


		""" Nodes indexes in the jetContent list for the pair that gives the max logLH (nodes to be removed and clustered)"""
		leftIdx = maxPairIdx[1]
		rightIdx = maxPairIdx[0]
		logger.debug(f" Left idx ={leftIdx}")
		logger.debug(f" Right idx  = {rightIdx}")


		""" Nodes indexes in the levelContent list for the pair that gives the max logLH (nodes to be removed and clustered)"""
		right = idx.index(rightIdx)
		left = idx.index(leftIdx)
		logger.debug(f" idx list = {idx}")


		""" Delete merged nodes """
		idx.pop(right)
		idx.pop(left)

		rightContent = levelContent.pop(right)
		leftContent = levelContent.pop(left)

		levelDeltas.pop(right)
		levelDeltas.pop(left)


		""" Delete nodes from all pairings list """
		sortPairs = [entry for entry in sortPairs
		             if (entry[1][0]!=leftIdx and entry[1][0]!=rightIdx and entry[1][1]!=leftIdx and entry[1][1]!=rightIdx )
		             ]


		""" Find new node pairings and insert into sortPairs list """

		newNode = np.sum([leftContent, rightContent], axis=0)
		newDelta = likelihood.get_delta_LR(leftContent, rightContent)

		NewNodePairs = [
				(
					likelihood.Basic_split_logLH(
						newNode,
						newDelta,
						levelContent[j],
						levelDeltas[j],
						delta_min,
						lam
					),
					[Nparent, idx[j]]
				)
				for j in range(len(idx))
			]

		# [bisect.insort(sortPairs, entry) for entry in NewNodePairs]

		sortPairs = sortPairs + NewNodePairs
		#
		# sortPairs = sorted(sortPairs, key=lambda x: x[0])
		dtype = [('logLH', float), ('pair', object)]
		a = np.array(sortPairs, dtype=dtype)
		sortPairs = np.sort(a, order='logLH')


		""" Update lists """

		idx.append(Nparent)

		levelContent.append(newNode)

		jetContent.append(newNode)

		levelDeltas.append(newDelta)

		N_leaves_list.append(N_leaves_list[leftIdx] + N_leaves_list[rightIdx])

		linkage_list.append([leftIdx, rightIdx, Nparent, N_leaves_list[-1]])

		jetTree.append([leftIdx, rightIdx])

		logLH.append(maxPairLogLH)


		""" Add updated path to predecessors list"""
		updatedPath = latentPath(
			path_sortPairs = sortPairs,
			path_levelContent=levelContent,
			path_levelDeltas=levelDeltas,
			path_logLH=logLH,
			path_jetTree=jetTree,
			path_jetContent=jetContent,
			path_idx=idx,
			path_N_leaves_list=N_leaves_list,
			path_linkage_list=linkage_list,
		)

		updatedPredecessors.append(updatedPath)


	return updatedPredecessors







