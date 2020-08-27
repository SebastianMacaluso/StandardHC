import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import logging
import copy

from scripts import linkageList
from scripts import reclusterTree
from scripts import beamSearchOptimal as beamSearch
from scripts.utils import get_logger

logger = get_logger(level=logging.INFO)


def HeatDendrogram(
		jet1 = None,
		jet2 = None,
		full_path = False,
		FigName = None,
):
	"""
	Create  a heat dendrogram clustermap.

	Args:
	:param truthJet: Truth jet dictionary
	:param recluster_jet1: reclustered jet 1
	:param recluster_jet2: reclustered jet 2
	:param full_path: Bool. If True, then use the total number of steps to connect a pair of leaves as the heat data. If False, then Given a pair of jet constituents {i,j} and the number of steps needed for each constituent to reach their closest common ancestor {Si,Sj}, the heat map scale represents the maximum number of steps, i.e. max{Si,Sj}.
	:param FigName: Dir and location to save a plot.
	"""

	# Build truth jet heat data
	Heatjet = copy.deepcopy(jet1)

	# if truth:
	# Calculate linkage list and add it to the truth jet dict
	linkageList.draw_truth(Heatjet)

	ancestors = Heatjet["tree_ancestors"]


	# Number of nodes from root to leaf for each leaf
	level_length = [len(entry) for entry in ancestors]
	max_level = np.max(level_length)

	# Pad tree_ancestors list for dim1=max_level, adding a different negative number at each row (for each leaf)
	ancestors1_array = np.asarray([np.concatenate(
		(ancestors[i], -(i + 1) * np.ones((max_level - len(ancestors[i]))))
	) for i in range(len(ancestors))])


	N_heat = len(ancestors1_array) # Number of constituents
	heat_data = np.zeros((N_heat, N_heat))
	neg_entries = np.sum(np.array(ancestors1_array) < 0, axis=1)

	#######################
	# Get total number of steps to connect a pair of leaves as the heat data
	if full_path:
		for i in range(N_heat):
			for j in range(i + 1, N_heat):

				logger.debug(f"Number of steps between nodes =  {np.count_nonzero(ancestors1_array[i]-ancestors1_array[j]==0)}")

				# Sum of number of nodes  from root to leaf for each leaf - 2 * number of nodes that are common ancestors
				heat_data[i, j] = (2 * max_level - (neg_entries[i] + neg_entries[j])) \
				                  - 2 * np.count_nonzero((ancestors1_array[i] - ancestors1_array[j]) == 0)

				heat_data[j, i] = heat_data[i, j]

				logger.debug(f"heat data = {heat_data[i, j]}")


	# Given a pair of jet constituents {i,j} and the number of steps needed for each constituent to reach their closest common ancestor
	# {Si,Sj}, the heat map scale represents the maximum number of steps, i.e. max{Si,Sj}.
	else:
		for i in range(N_heat):
			for j in range(i + 1, N_heat):

				logger.debug(f"Number of steps between nodes =  {np.count_nonzero(ancestors1_array[i]-ancestors1_array[j]==0)}")

				# Number of nodes for the longest path from root to leaf (between the 2 leaves) - Number of nodes that are common ancestors
				heat_data[i, j] = (max_level - np.minimum(neg_entries[i], neg_entries[j])) \
				                  - np.count_nonzero((ancestors1_array[i] - ancestors1_array[j]) == 0)

				heat_data[j, i] = heat_data[i, j]

				logger.debug(f"heat data = {heat_data[i, j]}")

	#######################
	# Build heat clustermap


	if not jet2:

		logger.info(f"{jet1['algorithm']} heat data ----  row: {jet1['algorithm']} -- alpha column: {jet1['algorithm']}")

		cm = sns.clustermap(
			heat_data,
			row_cluster=True,
			col_cluster=True,
			row_linkage=Heatjet["linkage_list"],
			col_linkage=Heatjet["linkage_list"],

		)

		if FigName:
			plt.savefig(str(FigName))

		# plt.show()

	else:
		logger.info(
			f"{jet1['algorithm']} heat data ----  row: {jet2['algorithm']} -- alpha column: {jet1['algorithm']}")

		cm = sns.clustermap(
			heat_data,
			row_cluster=True,
			col_cluster=True,
			row_linkage=jet2["linkage_list"],
			col_linkage=Heatjet["linkage_list"],
		)

		if FigName:
			plt.savefig(str(FigName))

		# plt.show()

	# ax = cm.fig.gca()
	ax = cm.ax_heatmap
	ax.set_xlabel(r"%s"%jet1['algorithm'], fontsize=15)

	if not jet2:
		ax.set_ylabel(r"%s"%jet1['algorithm'], fontsize=15)
	else:
		ax.set_ylabel(r"%s" % jet2['algorithm'], fontsize=15)


	ax.xaxis.set_label_coords(0.5, 1.3)
	ax.yaxis.set_label_coords(-0.3, 0.5)
	# ax.yaxis.labelpad = -1000
	plt.show()











def heat_dendrogram(
		truthJet = None,
		recluster_jet1 = None,
		recluster_jet2 = None,
		beamSearch_jet = None,
        beamSize = None,
		N_best = None,
		full_path = False,
		FigName = None,
):
	"""
	Create  a heat dendrogram clustermap.

	Args:
	:param truthJet: Truth jet dictionary
	:param recluster_jet1: reclustered jet 1
	:param recluster_jet2: reclustered jet 2
	:param full_path: Bool. If True, then use the total number of steps to connect a pair of leaves as the heat data. If False, then Given a pair of jet constituents {i,j} and the number of steps needed for each constituent to reach their closest common ancestor {Si,Sj}, the heat map scale represents the maximum number of steps, i.e. max{Si,Sj}.
	:param FigName: Dir and location to save a plot.
	"""

	# Build truth jet heat data
	if truthJet:

		# Calculate linkage list and add it to the truth jet dict
		linkageList.draw_truth(truthJet)

		ancestors = truthJet["tree_ancestors"]


	# Build jet 1 heat data
	else:

		if beamSearch_jet:

			# Calculate linkage list and add it to the truth jet dict
			linkageList.draw_truth(beamSearch_jet)

			ancestors = beamSearch_jet["tree_ancestors"]

			# reclustjet = beamSearch.recluster(beamSearch_jet,
			#                      delta_min=beamSearch_jet["pt_cut"],
			#                      lam=beamSearch_jet["Lambda"],
			#                      beamSize = beamSize,
			#                      N_best= N_best,
			#                      save=False)[0]
			# ancestors = reclustjet["tree_ancestors"]

		if recluster_jet1:
			# Recluster jet using itself as an input. This way, we use the constituents (leaves) as ordered in this jet and the tree_ancestors list for this algorithm. (Their leaves idx goes from 0 to N leaves in order when using jet 1 both as rows and colums)
			reclustjet = reclusterTree.recluster(recluster_jet1,
			                                     alpha=int(recluster_jet1["algorithm"]),
			                                     save=False)

			ancestors = reclustjet["tree_ancestors"]


	# Number of nodes from root to leaf for each leaf
	level_length = [len(entry) for entry in ancestors]
	max_level = np.max(level_length)

	# Pad tree_ancestors list for dim1=max_level, adding a different negative number at each row (for each leaf)
	ancestors1_array = np.asarray([np.concatenate(
		(ancestors[i], -(i + 1) * np.ones((max_level - len(ancestors[i]))))
	) for i in range(len(ancestors))])


	N_heat = len(ancestors1_array) # Number of constituents
	heat_data = np.zeros((N_heat, N_heat))
	neg_entries = np.sum(np.array(ancestors1_array) < 0, axis=1)

	#######################
	# Get total number of steps to connect a pair of leaves as the heat data
	if full_path:
		for i in range(N_heat):
			for j in range(i + 1, N_heat):

				logger.debug(f"Number of steps between nodes =  {np.count_nonzero(ancestors1_array[i]-ancestors1_array[j]==0)}")

				# Sum of number of nodes  from root to leaf for each leaf - 2 * number of nodes that are common ancestors
				heat_data[i, j] = (2 * max_level - (neg_entries[i] + neg_entries[j])) \
				                  - 2 * np.count_nonzero((ancestors1_array[i] - ancestors1_array[j]) == 0)

				heat_data[j, i] = heat_data[i, j]

				logger.debug(f"heat data = {heat_data[i, j]}")


	# Given a pair of jet constituents {i,j} and the number of steps needed for each constituent to reach their closest common ancestor
	# {Si,Sj}, the heat map scale represents the maximum number of steps, i.e. max{Si,Sj}.
	else:
		for i in range(N_heat):
			for j in range(i + 1, N_heat):

				logger.debug(f"Number of steps between nodes =  {np.count_nonzero(ancestors1_array[i]-ancestors1_array[j]==0)}")

				# Number of nodes for the longest path from root to leaf (between the 2 leaves) - Number of nodes that are common ancestors
				heat_data[i, j] = (max_level - np.minimum(neg_entries[i], neg_entries[j])) \
				                  - np.count_nonzero((ancestors1_array[i] - ancestors1_array[j]) == 0)

				heat_data[j, i] = heat_data[i, j]

				logger.debug(f"heat data = {heat_data[i, j]}")

	#######################
	# Build heat clustermap
	if truthJet: # ruth jet heat data

		if not recluster_jet1 and not beamSearch_jet:

			logger.info(f"truth heat data ----  alpha row: truth -- alpha column: truth")

			sns.clustermap(
				heat_data,
				row_cluster=True,
				col_cluster=True,
				row_linkage=truthJet["linkage_list"],
				col_linkage=truthJet["linkage_list"],
			)

			if FigName:
				plt.savefig(str(FigName))

			plt.show()

		elif recluster_jet1:

			logger.info(f"alpha row: {recluster_jet1['algorithm']} -- alpha column: truth")

			sns.clustermap(
				heat_data,
				row_cluster=True,
				col_cluster=True,
				row_linkage=recluster_jet1["linkage_list"],
				col_linkage=truthJet["linkage_list"],
			)

			if FigName:
				plt.savefig(str(FigName))

			plt.show()

		elif beamSearch_jet:

			logger.info(f"alpha row: {beamSearch_jet['algorithm']} -- alpha column: truth")

			sns.clustermap(
				heat_data,
				row_cluster=True,
				col_cluster=True,
				row_linkage=beamSearch_jet["linkage_list"],
				col_linkage=truthJet["linkage_list"],
			)

			if FigName:
				plt.savefig(str(FigName))

			plt.show()

	else: # jet 1 heat data

		if not recluster_jet2:
			logger.debug(f"reclustjet['linkage_list']= {reclustjet['linkage_list']}")
			logger.info(f"alpha row: {reclustjet['algorithm']} -- alpha column: {reclustjet['algorithm']}")

			sns.clustermap(
				heat_data,
				row_cluster=True,
				col_cluster=True,
				row_linkage=reclustjet["linkage_list"],
				col_linkage=reclustjet["linkage_list"],
			)
			if FigName:
				plt.savefig(str(FigName))

			plt.show()

		if recluster_jet2:

			reclustjet2 = reclusterTree.recluster(recluster_jet1, alpha=int(recluster_jet2["algorithm"]), save=False)

			logger.info(f"alpha row: {reclustjet2['algorithm']} -- alpha column: {reclustjet['algorithm']}")

			sns.clustermap(
				heat_data,
				row_cluster=True,
				col_cluster=True,
				row_linkage=reclustjet2["linkage_list"],
				col_linkage=reclustjet["linkage_list"],
				)

			if FigName:
				plt.savefig(str(FigName))

			plt.show()






def dendrogramDiff(
		truthJet = None,
		recluster_jet1 = None,
		recluster_jet2 = None,
		full_path = False,
		FigName = None,
):
	"""
	Create  a heat dendrogram displaying the difference between the clustermap.

	Args:
	:param truthJet: Truth jet dictionary
	:param recluster_jet1: reclustered jet 1
	:param recluster_jet2: reclustered jet 2
	:param full_path: Bool. If True, then use the total number of steps to connect a pair of leaves as the heat data. If False, then Given a pair of jet constituents {i,j} and the number of steps needed for each constituent to reach their closest common ancestor {Si,Sj}, the heat map scale represents the maximum number of steps, i.e. max{Si,Sj}.
	:param FigName: Dir and location to save a plot.
	"""

	#############################
	def getHeatMap(in_ancestors):
		# Number of nodes from root to leaf for each leaf
		level_length = [len(entry) for entry in in_ancestors]
		max_level = np.max(level_length)

		# Pad tree_ancestors list for dim1=max_level, adding a different negative number at each row (for each leaf)
		ancestors1_array = np.asarray([np.concatenate(
			(in_ancestors[i], -(i + 1) * np.ones((max_level - len(in_ancestors[i]))))
		) for i in range(len(in_ancestors))])

		N_heat = len(ancestors1_array) # Number of constituents
		heat_data = np.zeros((N_heat, N_heat))
		neg_entries = np.sum(np.array(ancestors1_array) < 0, axis=1)


		# Get total number of steps to connect a pair of leaves, as the heat data matrix
		if full_path:
			for i in range(N_heat):
				for j in range(i + 1, N_heat):

					logger.debug(f"Number of steps between nodes =  {np.count_nonzero(ancestors1_array[i]-ancestors1_array[j]==0)}")

					# Sum of number of nodes  from root to leaf for each leaf - 2 * number of nodes that are common ancestors
					heat_data[i, j] = (2 * max_level - (neg_entries[i] + neg_entries[j])) \
					                  - 2 * np.count_nonzero((ancestors1_array[i] - ancestors1_array[j]) == 0)

					heat_data[j, i] = heat_data[i, j]

					logger.debug(f"heat data = {heat_data[i, j]}")


		# Given a pair of jet constituents {i,j} and the number of steps needed for each constituent to reach their closest common ancestor
		# {Si,Sj}, the heat map scale represents the maximum number of steps, i.e. max{Si,Sj}.
		else:
			for i in range(N_heat):
				for j in range(i + 1, N_heat):

					logger.debug(f"Number of steps between nodes =  {np.count_nonzero(ancestors1_array[i]-ancestors1_array[j]==0)}")

					# Number of nodes for the longest path from root to leaf (between the 2 leaves) - Number of nodes that are common ancestors
					heat_data[i, j] = (max_level - np.minimum(neg_entries[i], neg_entries[j])) \
					                  - np.count_nonzero((ancestors1_array[i] - ancestors1_array[j]) == 0)

					heat_data[j, i] = heat_data[i, j]

					logger.debug(f"heat data = {heat_data[i, j]}")

		return heat_data

	#############################
	heat_data_jet1 = getHeatMap(recluster_jet1["tree_ancestors"])
	logger.debug(f"Jet 1 Heat_data = {heat_data_jet1}")

	new_heat_data_jet1 = heat_data_jet1[recluster_jet1["node_id"], :]
	logger.debug(f"Jet 1 Heat data after reordering the rows following the truth jet order {new_heat_data_jet1}")

	new_heat_data_jet1 = new_heat_data_jet1[:, recluster_jet1["node_id"]]
	logger.debug(f"Jet 1 Heat data after reordering the rows and columns following the truth jet order {new_heat_data_jet1}")


	if truthJet:

		# Calculate linkage list tree_ancestors list, and add them to the truth jet dict
		linkageList.draw_truth(truthJet)

		heat_data_truth= getHeatMap(truthJet["tree_ancestors"])
		logger.debug(f"Truth jet Heat_data = {heat_data_truth}")

		dataDiff = heat_data_truth - new_heat_data_jet1
		logger.info(f"(Truth jet - recluster jet1) heat data")

	elif recluster_jet2:

		heat_data_jet2 = getHeatMap(recluster_jet2["tree_ancestors"])

		new_heat_data_jet2 = heat_data_jet2[recluster_jet2["node_id"], :]
		logger.debug(f"Jet 2 Heat data after reordering the rows following the truth jet order {new_heat_data_jet2}")

		new_heat_data_jet2 = new_heat_data_jet2[:, recluster_jet2["node_id"]]
		logger.debug(f"Jet 2 Heat data after reordering the rows and columns following the truth jet order {new_heat_data_jet2}")

		dataDiff = new_heat_data_jet2 - new_heat_data_jet1

		logger.info(f"(recluster jet2 - recluster jet1) heat data")


	# Plot heat dendrogram differences
	sns.clustermap(
		dataDiff,
		row_cluster=False,
		col_cluster=False,
	)


	if FigName:
		plt.savefig(str(FigName))

	plt.show()






