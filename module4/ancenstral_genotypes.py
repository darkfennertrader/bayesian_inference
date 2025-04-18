from typing import List, Union
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.factors.discrete import DiscreteFactor


def genotypeGivenAlleleFreqsFactor(alleleFreqs, genotypeVarName: Union[str, int] = "G"):
    """
    Constructs a pgmpy DiscreteFactor for the probability distribution of a single
    genotype variable given allele frequencies.

    Parameters
    ----------
    alleleFreqs : list or 1D array
        The frequencies for each allele.  For example, [0.1, 0.9] for two alleles.

    genotypeVarName : str
        The name to use for the genotype variable in the factor.

    Returns
    -------
    factor : DiscreteFactor
        A pgmpy DiscreteFactor of dimension 1 (just the genotype variable).
        Its 'values' array has length M = (number_of_alleles choose 2) + number_of_alleles.
    """
    numAlleles = len(alleleFreqs)

    # Enumerate all genotypes (i, j) with i <= j.  We'll use zero-based indexing internally,
    # but you may adapt to 1-based if you want to match the Octave code exactly.
    genotypes = []
    for i in range(numAlleles):
        for j in range(i, numAlleles):
            genotypes.append((i, j))

    # Build the probabilities for each genotype
    card = len(genotypes)
    vals = np.zeros(card)
    for idx, (i, j) in enumerate(genotypes):
        p = alleleFreqs[i] * alleleFreqs[j]
        if i != j:
            p *= 2.0
        vals[idx] = p

    # Create the factor in pgmpy; it has 1 variable of cardinality = card
    factor = DiscreteFactor(
        variables=[genotypeVarName], cardinality=[card], values=vals
    )
    return factor


if __name__ == "__main__":
    # For two alleles with frequencies 0.1 and 0.9, we should get [0.01, 0.18, 0.81]
    alleleFreqs = np.array([0.1, 0.9])
    # alleleFreqs = np.array([0.1, 0.6, 0.3])
    factor = genotypeGivenAlleleFreqsFactor(alleleFreqs, genotypeVarName="GenotypeVar")
    print("Factor scope:", factor.scope())
    print("Factor cardinalities:", factor.cardinality)
    print("Factor values:", factor.values)
    # Expected output:
    # Factor values: [0.01 0.18 0.81]
