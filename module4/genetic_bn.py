from typing import List, Union, Dict, Optional
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.factors.discrete import DiscreteFactor
from ancenstral_genotypes import genotypeGivenAlleleFreqsFactor
from genotype_given_parents import genotypeGivenParentsGenotypesFactor
from not_mendelian_pheno_geno import phenotype_given_genotype_factor


FactorType = Union[DiscreteFactor, TabularCPD, Dict]


def construct_genetic_network(pedigree, alleleFreqs, alphaList):
    """
    Builds the factor list (length = 2*numPeople) for a pedigree with
    'numPeople' individuals.  For person i:
      - If pedigree.parents[i] == [0,0], then factorList[i] is the prior
        factor P(G_i) based on allele frequencies.
        Otherwise, factorList[i] is P(G_i | G_parent1, G_parent2).
      - factorList[numPeople + i] = P(Phenotype_i | Genotype_i).
    We number genotype variables as 1..n, phenotype variables as n+1..2n.
    """
    numPeople = len(pedigree["names"])
    factorList: List[Optional[FactorType]] = [None] * (2 * numPeople)
    numAlleles = len(alleleFreqs)

    for i in range(1, numPeople + 1):
        # i-th person's parents (1-based indexing). E.g. [0,0] means none.
        p1, p2 = pedigree["parents"][i - 1]
        if p1 == 0 and p2 == 0:
            # no parents => genotype prior
            factorList[i - 1] = genotypeGivenAlleleFreqsFactor(
                alleleFreqs, genotypeVarName=i
            )
        else:
            # has parents => genotypeGivenParentsGenotypesFactor
            factorList[i - 1] = genotypeGivenParentsGenotypesFactor(
                numAlleles, i, p1, p2
            )
        # phenotype factor always:
        factorList[numPeople + i - 1] = phenotype_given_genotype_factor(
            alphaList, genotype_var=i, phenotype_var=numPeople + i
        )
    return factorList


if __name__ == "__main__":
    pedigree = {"parents": [[0, 0], [1, 3], [0, 0]], "names": ["Ira", "James", "Robin"]}
    alleleFreqs = [0.1, 0.9]
    # For 2 alleles => 3 possible genotypes: (A1,A1), (A1,A2), (A2,A2).
    # alphaList[g] = P(Phenotype=has_trait | genotype=g). Must have length=3.
    alphaList = [0.8, 0.6, 0.1]

    factorList = construct_genetic_network(pedigree, alleleFreqs, alphaList)

    # Print the resulting factors (some are DiscreteFactor objects, some are dicts,
    # and the phenotype factors are TabularCPD objects):
    for idx, f in enumerate(factorList, start=1):
        print(f"Factor {idx}:")
        print(f)
