# import networkx as nx
# import matplotlib.pyplot as plt

# # 1. Create a directed graph
# G = nx.DiGraph()

# # 2. Add edges reflecting the same pedigree structure
# #    (Ira_G → Ira_P, Ira_G → James_G, Robin_G → Robin_P,
# #     Robin_G → James_G, James_G → James_P)
# edges = [
#     ("Ira_G", "Ira_P"),  # Ira's genotype → Ira's phenotype
#     ("Ira_G", "James_G"),  # Ira's genotype → James's genotype
#     ("Robin_G", "Robin_P"),  # Robin's genotype → Robin's phenotype
#     ("Robin_G", "James_G"),  # Robin's genotype → James's genotype
#     ("James_G", "James_P"),  # James's genotype → James's phenotype
# ]
# G.add_edges_from(edges)

# # Draw the graph
# plt.figure(figsize=(10, 6))
# pos = nx.spring_layout(G)  # positions for all nodes
# nx.draw(
#     G,
#     pos,
#     with_labels=True,
#     node_size=3000,
#     node_color="lightgreen",
#     font_size=10,
#     font_weight="bold",
#     arrowsize=20,
# )

# plt.title("Pedigree-Based Bayesian Network Structure")
# plt.savefig("04_pedigree_network.png", dpi=300)
# plt.show()

#####################################################################
from typing import List, Union
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD


def phenotype_given_genotype_factor(
    alpha_list: List[float],
    genotype_var: Union[str, int] = "Genotype",
    phenotype_var: Union[str, int] = "Phenotype",
) -> TabularCPD:
    """
    Builds a CPD for P(Phenotype | Genotype), where each genotype i
    has probability alpha_list[i] of "has trait."
    We assume 2 phenotype states ("has trait"=0, "no trait"=1) and n genotype states.
    """
    n = len(alpha_list)
    # First row = probabilities of having the trait
    # Second row = probabilities of not having the trait
    cpd_values = [
        [alpha_list[i] for i in range(n)],  # "has trait"
        [1.0 - alpha_list[i] for i in range(n)],  # "no trait"
    ]

    return TabularCPD(
        variable=phenotype_var,
        variable_card=2,
        values=cpd_values,
        evidence=[genotype_var],
        evidence_card=[n],
    )


if __name__ == "__main__":
    alphaList = [0.8, 0.6, 0.1]
    genotypeVar = "G"
    phenotypeVar = "F"

    # Build the CPD:
    cpd = phenotype_given_genotype_factor(alphaList, genotypeVar, phenotypeVar)
    print(cpd)

    # Check some values (by direct array indexing):
    print("P(hasTrait | Genotype=AA) =", cpd.values[0, 0])  # 0.8
    print("P(noTrait | Genotype=ff)  =", cpd.values[1, 2])  # 0.9
