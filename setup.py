import setuptools

setuptools.setup(
    name="StandardHC",
    version="0.0.1",
    description="Hierarchical Clustering Algorithms for Jet Physics. Includes:"
                "1) Likelihood based algorithms: Greedy and Beam Search"
                "2) Standard heuristic algorithms for jet physics such as kt, anti-kt and CA",
    url="https://github.com/SebastianMacaluso/StandardHC",
    author="Kyle Cranmer, Sebastian Macaluso, Duccio Pappadopulo",
    author_email="sm4511@nyu.edu",
    license="MIT",
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    # packages=setuptools.find_packages(),
    zip_safe=False,
)
