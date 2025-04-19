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

from typing import Union
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD


def phenotype_given_genotype_mendelian_factor(
    isDominant, genotypeVar: Union[str, int] = "G", phenotypeVar: Union[str, int] = "P"
):
    """
    Returns a pgmpy TabularCPD for P( phenotypeVar | genotypeVar ).

    We assume genotype has 3 states (1=AA, 2=Aa, 3=aa) and phenotype has 2 states
    (1=“has trait”, 2=“no trait”).  If isDominant=True, then AA or Aa => trait,
    aa => no trait; otherwise (isDominant=False) it is recessive (aa => trait).
    """
    if isDominant:
        # ‘Dominant’ table: columns=genotype=AA,Aa,aa; rows=phenotype=hasTrait,noTrait
        #   AA => 1.0 trait
        #   Aa => 1.0 trait
        #   aa => 0.0 trait
        cpd_values = [
            [1, 1, 0],  # P(phenotype=hasTrait | AA, Aa, aa)
            [0, 0, 1],  # P(phenotype=noTrait | AA, Aa, aa)
        ]
    else:
        # ‘Recessive’ table: columns=genotype=AA,Aa,aa; rows=phenotype=hasTrait,noTrait
        #   AA => 0.0 trait
        #   Aa => 0.0 trait
        #   aa => 1.0 trait
        cpd_values = [[0, 0, 1], [1, 1, 0]]

    return TabularCPD(
        variable=phenotypeVar,
        variable_card=2,
        values=cpd_values,
        evidence=[genotypeVar],
        evidence_card=[3],
    )


if __name__ == "__main__":
    isDominant = True
    genotypeVar = 1  # could be a string; here we just use an int
    phenotypeVar = 3  # likewise
    cpd = phenotype_given_genotype_mendelian_factor(
        isDominant, genotypeVar, phenotypeVar
    )
    print("TabularCPD for isDominant =", isDominant)
    print(cpd)

    # We can also check entries individually, e.g. "P(Phenotype=hasTrait | Genotype=AA)"
    # by indexing rows (phenotype), columns (genotype):
    # cpd.get_value(phenotype_state, genotype_state) uses 0-based indices internally:
    #
    #   genotype=AA is index 0 in these columns
    #   "has trait" is index 0 in the rows
    #
    # So the probability is:
    print("P(hasTrait|AA) = ", cpd.values[0, 0])
