import sys
import numpy as np
import logging
import pickle
import itertools
import copy


from scripts import likelihood

from scripts.utils import get_logger

logger = get_logger(level=logging.INFO)



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
			path_levelContent = None,
			path_levelDeltas = None,
			path_logLH = None,
			path_tree_dic = None,
			path_jetContent =None,
			path_idx = None,
			path_N_leaves_list = None,
			path_linkage_list = None,
	):

		self.levelContent = path_levelContent
		self.levelDeltas = path_levelDeltas
		self.logLH = path_logLH
		self.tree_dic = path_tree_dic
		self.jetContent =path_jetContent
		self.idx = path_idx
		self.N_leaves_list = path_N_leaves_list
		self.linkage_list = path_linkage_list






def recluster(
		input_jet,
		save = False,
		delta_min = None,
		lam = None,
		beamSize = None,
		N_best = None,
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
		- input_jet: any jet dictionary with the clustering history.

		- beamSize: beam size for the beam search algorithm, i.e. it determines the number of trees latent path run in parallel and kept in memory.

		- delta_min: pT cut scale for the showering process to stop.

		- lam: decaying rate value for the exponential distribution.

		- N_best = Number of jets that give the best log likelihood to be kept in memory

		- save: if true, save the reclustered jet dictionary list

	Returns:
		- jetsList: List of jet dictionaries
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
	jet_const = np.asarray(
	_rec(
	input_jet,
	-1,
	input_jet["root_id"],
	outers,
	)
	)


	bestLogLH_paths, \
	root_node = beamSearch(
		jet_const,
		delta_min = delta_min,
		lam = lam,
		beamSize = beamSize,
	)


	jetsList = []
	for path in bestLogLH_paths[-N_best::]:

		tree, \
		content, \
		node_id, \
		tree_ancestors = _traverse(root_node,
		                         path.jetContent,
		                         tree_dic=path.tree_dic,
		                         Nleaves=len(jet_const),
		                         )


		# Create jet dictionary with tree features
		jet = {}
		jet["root_id"] = 0
		jet["tree"] = np.asarray(tree).reshape(-1, 2)
		jet["content"] = np.asarray([np.asarray(c) for c in content]).reshape(-1, 2)
		jet["linkage_list"]=path.linkage_list
		jet["node_id"]=node_id
		jet["tree_ancestors"]=tree_ancestors
		jet["Nconst"]=len(jet_const)
		jet["algorithm"]= "beamSearch"
		jet["pt_cut"] = delta_min
		jet["Lambda"] = lam
		jet["logLH"] = path.logLH

		jetsList.append(jet)


	jetsList = jetsList[::-1]

	# Save reclustered tree
	if save:
		out_dir = "data/"

		algo = str(input_jet["name"]) + '_' + str(alpha)
		out_filename = out_dir + str(algo) + '.pkl'
		logger.info(f"Output jet filename = {out_filename}")
		with open(out_filename, "wb") as f:
			pickle.dump(jetsList, f, protocol=2)


	return jetsList







def beamSearch(
		constituents,
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

	Nconst = len(constituents)
	root_node = 2 * Nconst - 2
	logger.debug(f"Root node = (N constituents + N parent) = {root_node}")


	levelContent = np.asarray(constituents)
	levelDeltas = np.zeros(Nconst)
	logLH = []
	tree_dic = {}
	jetContent = copy.deepcopy(levelContent)
	idx = np.arange(Nconst)
	N_leaves_list = np.ones((Nconst))
	linkage_list = []

	# List that keeps track of all latent paths
	predecessors = [ ]

	path = latentPath(
		path_levelContent = levelContent,
		path_levelDeltas = levelDeltas,
		path_logLH=logLH,
		path_tree_dic = tree_dic,
		path_jetContent = jetContent,
		path_idx = idx,
		path_N_leaves_list=N_leaves_list,
		path_linkage_list=linkage_list,
	)

	predecessors.append(path)


	for level in range(len(levelContent) - 1):

		logger.debug(f"===============================================")
		logger.debug(f" LEVEL = {level}")
		logger.debug(f" LENGTH PREDECESSORS = {len(predecessors)}")

		total_levelLatentPaths = np.array([])

		for j in range(len(predecessors)):

			levelLatentPaths = level_SortedLogLH_beamPairs (
				in_levelContent = predecessors[j].levelContent,
				in_levelDeltas = predecessors[j].levelDeltas,
				delta_min = delta_min,
				lam = lam,
				beamSize = beamSize,
			)


			# Add total log likelihood for current latent path
			levelLatentPaths = [(x+np.sum(predecessors[j].logLH),y,x) for (x,y) in levelLatentPaths]


			# Append beam idx (it goes from 0 to beamSize) to list
			levelLatentPaths = np.insert(levelLatentPaths, 0, int(j), axis=1)
			logger.debug(f" levelLatentPaths = {levelLatentPaths}")


			# Append latent path to list containing the latent paths (the number of latent paths we keep is k with k = beamSize for each of the previous beam indexes  => we get a list of beamSize^2 latent paths)
			total_levelLatentPaths = np.append(total_levelLatentPaths, levelLatentPaths).reshape(-1, levelLatentPaths.shape[1])



		logger.debug(f" Lenght total_levelLatentPaths = {len(total_levelLatentPaths)}")

		# Sort all latent paths and keep the k ones with the biggest log likelihood (with k = beamSize)
		best_LevelLatentPaths = np.asarray(sorted(total_levelLatentPaths, key=lambda x: x[1])[-beamSize::])

		logger.debug(f" best_LevelLatentPaths = {best_LevelLatentPaths}")


		# Update latent paths and store them in predecessors
		predecessors = updateLevelPaths(
			best_LevelLatentPaths,
			prevPredecessors = predecessors,
			Nconst = Nconst,
			Nparent = level,
		)


	return predecessors, root_node








def level_SortedLogLH_beamPairs(
    in_levelContent = None,
	in_levelDeltas = None,
	delta_min = None,
	lam = None,
	beamSize = None,
):
	"""
	-Calculate all splitting log likelihood between all possible pair of constituents at a certain level and sort them.

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

	# Get all possible pairings
	pairs = np.asarray(list(itertools.combinations(np.arange(len(in_levelContent)), 2)))

	# Get all logLH at current level
	logLH_pairs = [(likelihood.split_logLH(in_levelContent[pairs][k][0], in_levelDeltas[pairs][k][0], in_levelContent[pairs][k][1],
	                       in_levelDeltas[pairs][k][1], delta_min, lam), k) for k in range(len(in_levelContent[pairs]))]

	maxPairs = sorted(logLH_pairs, key=lambda x: x[0])[-beamSize::]
	# logger.info(f" max paris = {np.asarray(maxPairs)}")

	return maxPairs







def updateLevelPaths(
		best_LevelPaths,
		prevPredecessors = None,
		Nconst= None,
		Nparent = None,
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

	# Get all possible pairings (all predecessor paths have the same number of nodes so we pick predecessor[0])
	pairs = np.asarray(list(itertools.combinations(np.arange(len(prevPredecessors[0].levelContent)), 2)))

	logger.debug(f"------------------------------------------------------------")

	updatedPredecessors = []

	for (beamIdx, sumLogLH, MaxPairIdx, pairlogLH) in best_LevelPaths:

		beamIdx = int(beamIdx)
		MaxPairIdx = int(MaxPairIdx)

		levelContent = copy.deepcopy(prevPredecessors[beamIdx].levelContent)
		levelDeltas = copy.deepcopy(prevPredecessors[beamIdx].levelDeltas)
		logLH = copy.deepcopy(prevPredecessors[beamIdx].logLH)
		N_leaves_list = copy.deepcopy(prevPredecessors[beamIdx].N_leaves_list)
		linkage_list = copy.deepcopy(prevPredecessors[beamIdx].linkage_list)
		tree_dic = copy.deepcopy(prevPredecessors[beamIdx].tree_dic)
		jetContent = copy.deepcopy(prevPredecessors[beamIdx].jetContent)
		idx = copy.deepcopy(prevPredecessors[beamIdx].idx)

		logger.debug(f" prevPredecessors[{beamIdx}].logLH  = {prevPredecessors[beamIdx].logLH}")
		logger.debug(f" Lenght prevPredecessors[{beamIdx}].logLH  = {len(prevPredecessors[beamIdx].logLH)}")


		# List that given a node idx, stores for that idx, the number of leaves for the branch below that node.
		N_leaves_list = np.concatenate(
			(N_leaves_list, [N_leaves_list[idx[pairs[MaxPairIdx][0]]] + N_leaves_list[idx[pairs[MaxPairIdx][1]]]])
		)

		# List with all the previous max log likelihood
		logger.debug(f" logLH before appending pair log LH = {logLH}")
		logLH = np.concatenate((logLH,[pairlogLH]))
		logger.debug(f" logLH after appending pair log LH = {logLH}")


		# Update linkage_list for the next level
		linkage_list.append([idx[pairs[MaxPairIdx][0]], idx[pairs[MaxPairIdx][1]], Nparent, N_leaves_list[-1]])


		# Update list of nodes momentum for the next level
		NewLevelContent = np.reshape(
		  np.append(np.delete(levelContent, pairs[MaxPairIdx], 0), [np.sum(levelContent[pairs[MaxPairIdx]], axis=0)]), (-1, 2))


		# Update list of Deltas for the next level
		NewLevelDeltas = np.append(np.delete(levelDeltas, pairs[MaxPairIdx], 0), [likelihood.get_delta_LR(levelContent[pairs[MaxPairIdx]][0], levelContent[pairs[MaxPairIdx]][1])])


		jetContent = np.concatenate((jetContent, [np.sum(levelContent[pairs[MaxPairIdx]], axis=0)]), axis=0)


		# Add a new key to the tree dictionary
		tree_dic[Nconst + Nparent] = idx[pairs[MaxPairIdx]]
		logger.debug(f" Nconst + Nparent =  {Nconst + Nparent}")
		logger.debug(f" MaxPairIdx = {MaxPairIdx}")
		logger.debug(f" idx = {idx}")
		logger.debug(f" pairs[MaxPairIdx] = {pairs[MaxPairIdx]}")
		logger.debug(f" idx[pairs[MaxPairIdx]] = {idx[pairs[MaxPairIdx]]}")


		# Delete the merged nodes
		logger.debug(f" Max pair = {pairs[MaxPairIdx]}")
		logger.debug(f" Lenght idx list = {len(idx)}")
		logger.debug(f" idx before adding new pair = {idx}")

		idx = np.concatenate((np.delete(idx, pairs[MaxPairIdx]), [Nconst + Nparent]), axis=0)

		logger.debug(f" idx after adding new pair = {idx}")
		logger.debug(f"  ")


		updatedPath = latentPath(
			path_levelContent = NewLevelContent,
			path_levelDeltas = NewLevelDeltas,
			path_logLH = logLH,
			path_tree_dic = tree_dic,
			path_jetContent = jetContent,
			path_idx = idx,
			path_N_leaves_list = N_leaves_list,
			path_linkage_list = linkage_list,
		)

		updatedPredecessors.append(updatedPath)


	return updatedPredecessors










def _traverse(
        root,
        jet_nodes,
        tree_dic=None,
        Nleaves=None,
        dendrogram=True,
):
    """
    This function call the recursive function _traverse_rec to make the trees starting from the root
    :param root: root node id
    :param jet_nodes: array with the momentum of all the nodes of the jet tree (both leaves and inners).
    :param tree_dic: dictionary that has the node id of a parent as a key and a list with the id of the 2 children as the values
    :param Nleaves: Number of constituents (leaves)
    :param dendrogram: bool. If True, then return tree_ancestors list.

    :return:
    - tree: Reclustered tree structure.
    - content: Reclustered tree momentum vectors
    - node_id:   list where leaves idxs are added in the order they appear when we traverse the reclustered tree (each number indicates the node id
    that we picked when we did the reclustering.). However, the idx value specifies the order in which the leaf nodes appear when traversing the origianl jet (e.g. truth level) jet . The value here is an integer between 0 and Nleaves.
    So if we went from truth to kt algorithm, then in the truth tree the leaves go as [0,1,2,3,4,,...,Nleaves-1]
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
    jet_nodes,
    tree_dic=tree_dic,
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
        jet_nodes,
        tree_dic=None,
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
    :param jet_nodes: array with the momentum of all the nodes of the jet tree (both leaves and inners).
    :param tree_dic: dictionary that has the node id of a parent as a key and a list with the id of the 2 children as the values
    :param Nleaves: Number of constituents (leaves)
    :param node_id: list where leaves idxs are added in the order they appear when we traverse the reclustered tree (each number indicates the node id
    that we picked when we did the reclustering.). However, the idx value specifies the order in which the leaf nodes appear when traversing the truth level jet . The value here is an integer between 0 and Nleaves.
    So if we went from truth to kt algorithm, then in the truth tree the leaves go as [0,1,2,3,4,,...,Nleaves-1]
    :param ancestors: 1 entry of tree_ancestors (there is one for each leaf of the tree). It is appended to tree_ancestors.
    :param tree_ancestors: List with one entry for each leaf of the tree, where each entry lists all the ancestor node ids when traversing the tree from the root to the leaf node.
    :param dendrogram: bool. If True, append ancestors to tree_ancestors list.
    """


    id = len(tree) // 2
    if parent_id >= 0:
        if is_left:
            tree[2 * parent_id] = id  # We set the location of the lef child in the content array. So the left child momentum will be content[tree[2 * parent_id]]
        else:
            tree[2 * parent_id + 1] = id  # We set the location of the right child in the content array. So the right child will be content[tree[2 * parent_id+1]]
    """"(With each 4-vector we increase the content array by one element and the tree array by 2 elements. But then we take id=tree.size()//2, so the id increases by 1. The left and right children are added one after the other.)"""


    # Insert 2 new nodes to the vector that constitutes the tree.
    # In the next iteration we will replace this 2 values with the location of the parent of the new nodes
    tree.append(-1)
    tree.append(-1)

    # Fill the content vector with the values of the node
    content.append(jet_nodes[root])

    new_ancestors = None
    if dendrogram:
        new_ancestors = copy.deepcopy(ancestors)
        logger.debug(f" ancestors before = {ancestors}")
        new_ancestors = np.append(new_ancestors, root)  # Node ids in terms of the truth jet dictionary
        logger.debug(f" ancestors after = {ancestors}")


    # We move from the root down until we get to the leaves. We do this recursively
    if root >= Nleaves:

        children = tree_dic[root]

        # print(" root = ", root)
        # print(" Children = ",children)


        logger.debug(f"Children = {children}")

        L_idx = children[0]
        R_idx = children[1]


        _traverse_rec(L_idx, id,
                      True,
                      tree,
                      content,
                      jet_nodes,
                      tree_dic,
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
                      jet_nodes,
                      tree_dic,
                      Nleaves=Nleaves,
                      node_id=node_id,
                      ancestors=new_ancestors,
                      dendrogram=dendrogram,
                      tree_ancestors=tree_ancestors,
                      )

    # If not then its a leaf
    else:
        node_id.append(root)
        if dendrogram:
            tree_ancestors.append(new_ancestors)
            logger.debug(f"tree_ancestors= {tree_ancestors}")










def getConstituents(jet, parent, node_id, outers_list):
	"""
	Recursive function to get a list of the tree leaves
	"""
	if jet["tree"][node_id, 0] == -1:

		outers_list.append(jet["content"][node_id])

	else:
		getConstituents(
	    jet,
	    node_id,
	    jet["tree"][node_id, 0],
	    outers_list,
	    )

		getConstituents(
	    jet,
	    node_id,
	    jet["tree"][node_id, 1],
	    outers_list,
	    )

	return outers_list


