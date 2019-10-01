import sys
import numpy as np
import logging
import pickle
import itertools
import time
import copy

from scripts import likelihood
from scripts.utils import get_logger

logger = get_logger(level=logging.INFO)


"""
This is an O(N^2) algorithm for a greedy clustering of nodes into a tree, based on the maximum likelihood. The algorithm also builds a dictionary with features needed to traverse, access nodes info and visualize the clustered trees.
"""


def recluster(
		input_jet,
		save = False,
		delta_min = None,
		lam = None,
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
		- jet["algorithm"]: Algorithm to generate the tree structure, e.g. truth, kt, antikt, CA, likelihood.

	Args:
		- input_jet: any jet dictionary with the clustering history.

		- delta_min: pT cut scale for the showering process to stop.

		- lam: decaying rate value for the exponential distribution.

		- save: if true, save the reclustered jet dictionary list

		- visualize: if true, calculate extra features needed for the visualizations and add them to the tree dictionary.

	Returns:
		- jet dictionary
	"""


	def _rec(jet, parent, node_id, outers_list):
		"""
        Recursive function to get a list of the tree leaves
        """
		if jet["tree"][node_id, 0] == -1:

			outers_list.append(jet["content"][node_id])

		else:
			_rec(
		    jet,
		    node_id,
		    jet["tree"][node_id, 0],
		    outers_list,
		    )

			_rec(
		    jet,
		    node_id,
		    jet["tree"][node_id, 1],
		    outers_list,
		    )

		return outers_list


	outers = []

	# Get constituents list (leaves)
	jet_const =_rec(
	input_jet,
	-1,
	input_jet["root_id"],
	outers,
	)


	start_time = time.time()

	# Run clustering algorithm
	tree, \
	idx, \
	jetContent, \
	root_node, \
	Nconst, \
	N_leaves_list, \
	linkage_list,\
	logLH = greedyLH(
		jet_const,
		delta_min = delta_min,
		lam = lam,
		M_Hard = input_jet["M_Hard"],
	)


	jet = {}

	""" Extra features needed for visualizations """
	if visualize:
		tree,\
		jetContent,\
		node_id,\
		tree_ancestors  = _traverse(
			root_node,
		    jetContent,
		    jetTree = tree,
		    Nleaves = Nconst,
		)

		jet["node_id"]=node_id
		jet["tree_ancestors"]=tree_ancestors
		root_node = 0


	jet["root_id"] = root_node
	jet["tree"] = np.asarray(tree).reshape(-1, 2)
	jet["content"] = np.asarray(jetContent).reshape(-1, 2)
	jet["linkage_list"]=linkage_list
	jet["Nconst"]=Nconst
	jet["algorithm"]= "greedyLH"
	jet["pt_cut"] = delta_min
	jet["Lambda"] = lam
	jet["logLH"] = np.asarray(logLH)


	logger.debug(f" Recluster and build tree algorithm total time = {time.time()  - start_time}")


	""" Save reclustered tree """
	if save:
		out_dir = "data/"
		logger.info(f"input_jet[name]= {input_jet['name']}")

		algo = str(input_jet["name"]) + '_' + str(alpha)
		out_filename = out_dir + str(algo) + '.pkl'
		logger.info(f"Output jet filename = {out_filename}")
		with open(out_filename, "wb") as f:
			pickle.dump(jet, f, protocol=2)


	return jet







def getConstituents(jet, node_id, outers_list):
	"""
	Recursive function to get a list of the tree leaves
	"""
	if jet["tree"][node_id, 0] == -1:

		outers_list.append(jet["content"][node_id])

	else:
		getConstituents(
	    jet,
	    jet["tree"][node_id, 0],
	    outers_list,
	    )

		getConstituents(
	    jet,
	    jet["tree"][node_id, 1],
	    outers_list,
	    )

	return outers_list


def greedyLH(levelContent, delta_min= None, lam=None, M_Hard = None):
	"""
	Runs the logLHMaxLevel function level by level starting from the list of constituents (leaves) until we reach the root of the tree.

	Note: levelContent is a list of the nodes after deleting the constituents that are merged and adding the new node in each level.
	      So this should only have the root of the tree at the end.

	Args:
		- levelContent: jet constituents (i.e. the leaves of the tree)
		- delta_min: pT cut scale for the showering process to stop.
		- lam: decaying rate value for the exponential distribution.


	Returns:

	  - jetTree: dictionary that has the node id of a parent as a key and a list with the id of the 2 children as the values
	  - idx: array that stores the node id
	   (the node id determines the location of the momentum vector of a node in the jetContent array)
	    of the nodes that are in the current levelContent array. It has the same elements as the content_level (they get updated
	    level by level).
	  - jetContent: array with the momentum of all the nodes of the jet tree (both leaves and inners).
	  - root_node: root node id
	  - Nconst: Number of leaves of the jet
	  - N_leaves_list: List that given a node idx, stores for that idx, the number of leaves for the branch below that node.
	  - linkage_list: linkage list to build heat clustermap visualizations.
	  - logLH: list with the log likelihood of each pairing.
	"""

	Nconst = len(levelContent)

	jetTree = [[-1,-1]]*Nconst
	idx = [i for i in range(Nconst)]
	jetContent = copy.deepcopy(levelContent)
	root_node = 2 * Nconst - 2
	N_leaves_list = [1.] * Nconst
	linkage_list = []
	logLH=[]
	levelDeltas = [0.] * Nconst


	""" Calculate the nearest neighbor (NN) based on max log likelihood (logLH) for each leaf of the tree."""
	NNpairs = NNeighbors(
			levelContent,
			levelDeltas,
			Nconst = Nconst,
			delta_min = delta_min,
			lam = lam,
	)


	""" Cluster constituents. This is O(N) at each level x N levels => O(N^2) """
	for j in range(Nconst - 1):

		logger.debug(f"===============================================")
		logger.debug(f" LEVEL = {j}")

		logLHMaxLevel(
			NNpairs,
			levelContent,
			levelDeltas,
			logLH,
			jetTree,
			jetContent,
			idx,
			Nparent = Nconst + j,
			N_leaves_list = N_leaves_list,
			linkage_list = linkage_list,
			delta_min = delta_min,
			lam = lam,
		)

	# logger.info(f" log LH list before setting last merging likelihood to 1 = {logLH}")
	if M_Hard is not None:

		logLH[-1]=0

	# logger.info(f" log LH list after setting last merging likelihood to 1 = {logLH}")

	return jetTree, idx, jetContent, root_node, Nconst, N_leaves_list, linkage_list, logLH







def NNeighbors(
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
		- NNpairs

	"""

	NNpairs =  [(-np.inf, [0, -999])] + \
	           [
		           max(
			           [
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
				           for j in range(k, 0, -1)
			           ], key=lambda x: x[0]
		           )
		           for k in range(1, Nconst, 1)
	           ]


	return NNpairs







def logLHMaxLevel(
	NNpairs,
    levelContent,
	levelDeltas,
    logLH,
    jetTree,
    jetContent,
    idx,
    Nparent=None,
    N_leaves_list=None,
	linkage_list=None,
	delta_min = None,
	lam = None,
):
	"""
	- Update the jet dictionary information by deleting the nodes that are merged and adding the new node at each level.
	- Update the nearest neighbors (NN) list of the nodes whose NN was deleted.

	Args:

		- levelContent: array with the constituents momentum list for the current level (i.e. after deleting the constituents that are merged and
	      adding the new node from merging them in all previous levels)
	    - levelDeltas: array with the delta values (for the splitting of a node in the Toy Jets Shower Model) for the current level.
	    - logLH: list with all the previous max log likelihood pairings.
	    - jetTree: jet tree structure list
	    - jetContent: array with the momentum of all the nodes of the jet tree (both leaves and inners) after adding one
	      more level in the clustering.
	      (We add a new node each time we cluster 2 pseudojets)
	    - idx: array that stores the node id (the node id determines the location of the momentum of a node in the jetContent array)
	      of the nodes that are in the current levelContent array. It has the same number of elements as levelContent (they get updated
	      level by level).
	    - Nparent: index of each parent added to the tree.
		- N_leaves_list: List that given a node idx, stores for that idx, the number of leaves for the branch below that node. It is initialized only with the tree leaves
	    - linkage_list: linkage list to build heat clustermap visualizations.
	      [SciPy linkage list website](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html)
	      Linkage list format: A  (n - 1) by 4 matrix Z is returned. At the i-th iteration, clusters with indices Z[i, 0] and Z[i, 1] are combined to form cluster (n + 1) . A cluster with an index less than n  corresponds to one of the n original observations. The distance between clusters Z[i, 0] and Z[i, 1] is given by Z[i, 2]. The fourth value Z[i, 3] represents the number of original observations in the newly formed cluster.
	    - delta_min: pT cut scale for the showering process to stop.
		- lam: decaying rate value for the exponential distribution.

	"""


	""" Index of the pair that gives the max logLH, also the index of the right node to be removed """

	logger.debug(f" maxPairLogLH, maxPairIdx = {max(NNpairs, key=lambda x: x[0])}")
	logger.debug(f" NNpairs = {NNpairs}")
	right = NNpairs.index(max(NNpairs, key=lambda x: x[0]))


	""" Remove nodes pair with max logLH """
	maxPairLogLH, maxPairIdx = NNpairs.pop(right)

	leftIdx = maxPairIdx[1]
	rightIdx = maxPairIdx[0]


	""" Index of the left node to be removed """
	left = [entry[1][0] for entry in NNpairs].index(leftIdx)
	logger.debug(f" left idxs list = {[entry[1][0] for entry in NNpairs]}")

	NNpairs.pop(left)



	""" Update levelDeltas, idx, levelContent, jetContent, N_leaves_list, linkage_list, jetTree and logLH lists """
	idx.pop(right)
	idx.pop(left)
	idx.append(Nparent)

	rightContent = levelContent.pop(right)
	leftContent = levelContent.pop(left)

	newNode = np.sum([leftContent,rightContent],axis = 0)
	levelContent.append(newNode)
	jetContent.append(newNode)

	newDelta = likelihood.get_delta_LR(leftContent,rightContent)
	levelDeltas.pop(right)
	levelDeltas.pop(left)
	levelDeltas.append(newDelta)

	N_leaves_list.append(N_leaves_list[leftIdx] + N_leaves_list[rightIdx])

	linkage_list.append([leftIdx, rightIdx, Nparent, N_leaves_list[-1]])

	jetTree.append([leftIdx, rightIdx])

	logLH.append(maxPairLogLH)



	""" Find if any other node had one of the merged nodes as its NN """
	NNidxUpdate = [i for i, entry in enumerate(NNpairs) if (entry[1][1] == leftIdx or entry[1][1] == rightIdx)]

	if NNidxUpdate!=[]:

		logger.debug(f" Indices that need to get the NN updated = {NNidxUpdate}")
		logger.debug(f" First entry of NNpairs = {NNpairs[0]}")

		if NNidxUpdate[0]==0:
			NNpairs[NNidxUpdate[0]] = (-np.inf, NNpairs[NNidxUpdate[0]][1])
			NNidxUpdate = NNidxUpdate[1::]

		NNpairsUpdate = [
			max(
				[
					(
						likelihood.Basic_split_logLH(
							levelContent[k],
							levelDeltas[k],
							levelContent[k - j],
							levelDeltas[k - j],
							delta_min,
							lam
						),
						[idx[k], idx[k - j]]
					)
					for j in range(k, 0, -1)
				],
				key=lambda x: x[0]
			)
			for k in NNidxUpdate
		]

		for i,entry in enumerate(NNidxUpdate):
			NNpairs[entry] = NNpairsUpdate[i]



	""" Find merged node NN and append to list """
	NewNodeNN = max(
		[
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
			for j in range(len(levelContent))
		],
		key=lambda x: x[0]
	)

	NNpairs.append(NewNodeNN)










def _traverse(
        root,
        jetContent,
        jetTree=None,
        Nleaves=None,
        dendrogram=True,
):
	"""
    This function call the recursive function _traverse_rec to make the trees starting from the root
    :param root: root node id
    :param jetContent: array with the momentum of all the nodes of the jet tree (both leaves and inners).
    :param jetTree: dictionary that has the node id of a parent as a key and a list with the id of the 2 children as the values
    :param Nleaves: Number of constituents (leaves)
    :param dendrogram: bool. If True, then return tree_ancestors list.

    :return:
    - tree: Reclustered tree structure.
    - content: Reclustered tree momentum vectors
    - node_id:   list where leaves idxs are added in the order that they appear when we traverse the reclustered tree (each number indicates the node id that we picked when we did the reclustering.). However, the idx value specifies the order in which the leaf nodes appear when traversing the origianl jet (e.g. truth level jet). The value here is an integer between 0 and Nleaves.
    So if we go from truth to kt algorithm, then in the truth tree the leaves go as [0,1,2,3,4,,...,Nleaves-1]
    - tree_ancestors: List with one entry for each leaf of the tree, where each entry lists all the ancestor node ids when traversing the tree from the root to the leaf node.

    """

	tree = []
	content = []
	node_id = []
	tree_ancestors = []

	_traverse_rec(
    root,
    -1,
    False,
    tree,
    content,
    jetContent,
    jetTree=jetTree,
    Nleaves=Nleaves,
    node_id=node_id,
    ancestors=[],
    tree_ancestors=tree_ancestors,
    dendrogram=dendrogram,
    )

	return tree, content, node_id, tree_ancestors






def _traverse_rec(
        root,
        parent_id,
        is_left,
        tree,
        content,
        jetContent,
        jetTree=None,
        Nleaves=None,
        node_id=None,
        ancestors=None,
        tree_ancestors=[],
        dendrogram=False,
):
	"""
	Recursive function to build the jet tree structure.
	:param root: parent node momentum
	:param parent_id: parent node idx
	:param is_left: bool.
	:param tree: List with the tree
	:param content: List with the momentum vectors
	:param jetContent: array with the momentum of all the nodes of the jet tree (both leaves and inners).
	:param jetTree: dictionary that has the node id of a parent as a key and a list with the id of the 2 children as the values
	:param Nleaves: Number of constituents (leaves)
	:param node_id: list where leaves idxs are added in the order they appear when we traverse the reclustered tree (each number indicates the node id
	that we picked when we did the reclustering.). However, the idx value specifies the order in which the leaf nodes appear when traversing the truth level jet . The value here is an integer between 0 and Nleaves.
	So if we went from truth to kt algorithm, then in the truth tree the leaves go as [0,1,2,3,4,,...,Nleaves-1]
	:param ancestors: 1 entry of tree_ancestors (there is one for each leaf of the tree). It is appended to tree_ancestors.
	:param tree_ancestors: List with one entry for each leaf of the tree, where each entry lists all the ancestor node ids when traversing the tree from the root to the leaf node.
	:param dendrogram: bool. If True, append ancestors to tree_ancestors list.
	"""

	""""
	(With each momentum vector we increase the content array by one element and the tree array by 2 elements. 
	But then we take id=tree.size()//2, so the id increases by 1.)
	"""
	id = len(tree) // 2

	if parent_id >= 0:
		if is_left:

			"""Insert in the tree list, the location of the lef child in the content array."""
			tree[2 * parent_id] = id
		else:

			"""Insert in the tree list, the location of the right child in the content array."""
			tree[2 * parent_id + 1] = id


	"""Insert 2 new nodes to the vector that constitutes the tree. If the current node is a parent, then we will replace the -1 with its children idx in the content array"""
	tree.append(-1)
	tree.append(-1)


	""" Append node momentum to content list """
	content.append(jetContent[root])


	""" Fill node ancestors list """
	new_ancestors = None
	if dendrogram:
		new_ancestors = np.copy(ancestors)
		logger.debug(f" ancestors before = {ancestors}")

		new_ancestors = np.append(new_ancestors, root)
		logger.debug(f" ancestors after = {ancestors}")


	""" Move from the root down recursively until we get to the leaves. """
	if root >= Nleaves:

		children = jetTree[root]

		logger.debug(f"Children = {children}")

		L_idx = children[0]
		R_idx = children[1]


		_traverse_rec(L_idx, id,
                      True,
                      tree,
                      content,
                      jetContent,
                      jetTree,
                      Nleaves=Nleaves,
                      node_id=node_id,
                      ancestors=new_ancestors,
                      dendrogram=dendrogram,
                      tree_ancestors=tree_ancestors,
                      )

		_traverse_rec(R_idx,
                      id,
                      False,
                      tree,
                      content,
                      jetContent,
                      jetTree,
                      Nleaves=Nleaves,
                      node_id=node_id,
                      ancestors=new_ancestors,
                      dendrogram=dendrogram,
                      tree_ancestors=tree_ancestors,
                      )



	else:
		""" If the node is a leaf, then append idx to node_id and its ancestors as a new row of tree_ancestors """

		node_id.append(root)

		if dendrogram:

			tree_ancestors.append(new_ancestors)
			logger.debug(f"tree_ancestors= {tree_ancestors}")



