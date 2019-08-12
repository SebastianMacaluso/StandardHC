# Binary Tree Reclustering Algorithms for Jets Physics

### **Kyle Cranmer, Sebastian Macaluso and Duccio Pappadopulo**

Note that this is an early development version. 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

<!--## Introduction-->
<!---->
<!--This package implements different algorithms to recluster a set of jet constituents (leaves) into a binary tree.-->
<!---->
<!-- In particular, this package explores how different algorithms can reconstruct the latent structure of the jets generated with the [`Toy Generative Model for Jets`](https://github.com/SebastianMacaluso/ToyJetsShower) package. Comparisons and visualizations are enables thanks to the   [`VisualizeBinaryTrees`](https://github.com/SebastianMacaluso/VisualizeBinaryTrees) package.-->
 
 ## Motivation and problem formulation
 
 
 At physics colliders, e.g. the Large Hadron Collider at CERN, two beams of particles (accelerated to very high energies) directed against each other are collided head-on. As a result, new unstable particles get created and *showering process* happens, where successive binary splittings of these initial unstable particles are produced until all the particles are stable, e.g. . This process gives rise to jets, which are a collimated spray of energetic charged and neutral particles. We refer to these final particles as the jet constituents.
 
<!-- Determining the nature or type of the initial unstable particle (and its children and grandchildren) that gave rise to a specific jet is essential in searches of new physics as well as precission measurements of the current model, i.e. the Standard Model of Particle Physics.-->
 
  As a result of this *showering process*, there could be many latent paths that may lead to a specific jet (i.e. the set of constituents). Thus, it is natural to represent a jet and the particular showering path that gave rise to it as a binary tree, where the inner nodes represent each of the unstable particles and the leaves represent the jet constituents.   
 
 
 In this context, it becomes relevant and interesting to study algorithms to reconstruct the jet constituents (leaves) into a binary tree and how close these algorithms can reconstruct the truth latent path. Being able to perform a precise reconstruction of the truth tree would assist in physcis searches at the Large Hadron Collider. In particular, determining the nature or type of the initial unstable particle (and its children and grandchildren) that gave rise to a specific jet is essential in searches of new physics as well as precission measurements of the current model, i.e. the Standard Model of Particle Physics.
 
 There are software tools called **Parton Showers** that encode a physics model for the simulation of jets that are produced at colliders.
  A python package for a generative model to aid in machine learning (ML) research for jet physics was provided in [`Toy Generative Model for Jets`](https://github.com/SebastianMacaluso/ToyJetsShower). This model has a tractable likelihood, and is as simple and easy to describe as possible but at the same time captures the essential ingredients of parton shower generators in full physics simulations.
 
 ## Reclustering Algorithms

 This package implements different algorithms to recluster a set of jet constituents (leaves) into a binary tree.
 In particular, we explore how different algorithms can reconstruct the latent structure of the jets generated with the [`Toy Generative Model for Jets`](https://github.com/SebastianMacaluso/ToyJetsShower) package.  Comparisons and visualizations are enabled thanks to the   [`VisualizeBinaryTrees`](https://github.com/SebastianMacaluso/VisualizeBinaryTrees) package, which is also included within this package.
 
 
 There can be many techniques to recluster the set of jet constituents (leaves) into a binary tree, from machine learning based to more traditional algorithms.
 
 
<!-- Comparisons and visualizations are enables thanks to the   [`VisualizeBinaryTrees`](https://github.com/SebastianMacaluso/VisualizeBinaryTrees) package.-->
<!-- -->
<!-- Comparisons and visualizations are enables thanks to the   [`VisualizeBinaryTrees`](https://github.com/SebastianMacaluso/VisualizeBinaryTrees) package.-->
 
<!-- -->
<!-- package.-->
<!-- we provide a standalone description of a generative model to aid in machine learning (ML) research for jet physics. The motivation is to build a model that has a tractable likelihood, and is as simple and easy to describe as possible but at the same time captures the essential ingredients of parton shower generators in full physics simulations.-->
<!-- -->
<!--  Parton shower generators are software tools that encode a physics model for the simulation of jets that are produced at colliders, e.g. the Large Hadron Collider at CERN.-->
 
 
<!-- If the truth jet tree structure is known,-->
<!-- The closer the tree reconstructed tree is to the -->
<!-- -->
<!-- -->
<!-- Thus, starting from the initial unstable particle, successive splittings are produced until all the particles are stable (i.e. the stopping rule is satisfied for each of the final particles). We refer to this final particles as the jet constituents.-->
<!-- -->
<!-- -->
<!-- In this notes, we provide a standalone description of a generative model to aid in machine learning (ML) research for jet physics. The motivation is to build a model that has a tractable likelihood, and is as simple and easy to describe as possible but at the same time captures the essential ingredients of parton shower generators in full physics simulations.  The aim is for the model to have a python implementation with few software dependencies.-->
<!-- -->
<!-- Parton shower generators are software tools that encode a physics model for the simulation of jets that are produced at colliders, e.g. the Large Hadron Collider at CERN.-->
<!-- Jets are a collimated spray of energetic charged and neutral particles. Parton showers generate the particle content of a jet, going through a cascade process, starting from an initial unstable particle. In this description, there is a recursive algorithm that produces binary splittings of an unstable parent particle into two children particles, and a stopping rule. Thus, starting from the initial unstable particle, successive splittings are implemented until all the particles are stable (i.e. the stopping rule is satisfied for each of the final particles). We refer to this final particles as the jet constituents.-->
<!-- -->
<!-- As a result of this {\it showering process}, there could be many latent paths that may lead to a specific jet (i.e. the set of constituents). Thus, it is natural and straightforward to represent a jet and the particular showering path that gave rise to it as a binary tree, where the inner nodes represent each of the unstable particles and the leaves represent the jet constituents.  -->
 







##### **Running the simulation locally as a python package:**

1. Clone the *ReclusterTreeAlgorithms* repository
2. `cd ReclusterTreeAlgorithms`
3. `make`


<pre>



</pre>

<img src="https://github.com/SebastianMacaluso/ReclusterTreeAlgorithms/blob/master/plots/IRIS-HEP.png" width="300" align="left"> <img src="https://github.com/SebastianMacaluso/ReclusterTreeAlgorithms/blob/master/plots/NYU.png" width="200" align="center"> <img src="https://github.com/SebastianMacaluso/ReclusterTreeAlgorithms/blob/master/plots/MSDSE.png" width="300" align="right">






