from typing import List, Union
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.factors.discrete import DiscreteFactor


def generateAlleleGenotypeMappers(numAlleles):
    """
    Constructs the same 1-based mappings used by the course code:

    - allelesToGenotypes[a,b] = genotype ID for alleles (a,b)
    - genotypesToAlleles[g,:] = the two allele IDs for genotype g

    where a, b, g all run from 1..(numAlleles) or 1..(number_of_genotypes).
    """
    # Number of distinct genotypes = (numAlleles choose 2) + numAlleles
    m = (numAlleles * (numAlleles - 1)) // 2 + numAlleles

    # We will store these in 1-based arrays for consistency with the MATLAB code.
    allelesToGenotypes = np.zeros((numAlleles + 1, numAlleles + 1), dtype=int)
    genotypesToAlleles = np.zeros((m + 1, 2), dtype=int)

    genotypeID = 1
    for a in range(1, numAlleles + 1):
        for b in range(a, numAlleles + 1):
            genotypesToAlleles[genotypeID, 0] = a
            genotypesToAlleles[genotypeID, 1] = b
            # Make it symmetric
            allelesToGenotypes[a, b] = genotypeID
            allelesToGenotypes[b, a] = genotypeID
            genotypeID += 1

    return allelesToGenotypes, genotypesToAlleles


def genotypeGivenParentsGenotypesFactor(
    numAlleles, genotypeVarChild, genotypeVarParentOne, genotypeVarParentTwo
):
    """
    Translates the Octave 'genotypeGivenParentsGenotypesFactor.m' into Python.

    Returns a Python dict with fields:
       .var  = [childVar, parentVar1, parentVar2]
       .card = [m, m, m]   (where m is the number of genotypes)
       .val  = flattened CPD table, in child-fastest order.
    """
    # Build the allele↔genotype maps:
    allelesToGenotypes, genotypesToAlleles = generateAlleleGenotypeMappers(numAlleles)

    # Number of possible genotypes:
    m = (numAlleles * (numAlleles - 1)) // 2 + numAlleles

    # We store probabilities in a 3D array, 1-based for each dimension:
    # dimension: [child_genotype, parent1_genotype, parent2_genotype]
    cpvals = np.zeros((m + 1, m + 1, m + 1))

    # Fill in the counts for each combination of parents' genotypes:
    for i in range(1, m + 1):
        ia = genotypesToAlleles[i]  # e.g. [1,1], [1,2], or [2,2]
        for j in range(1, m + 1):
            ja = genotypesToAlleles[j]
            # The child genotype can come from any of the 4 parental-allele pairs:
            child1 = allelesToGenotypes[ia[0], ja[0]]
            child2 = allelesToGenotypes[ia[0], ja[1]]
            child3 = allelesToGenotypes[ia[1], ja[0]]
            child4 = allelesToGenotypes[ia[1], ja[1]]
            cpvals[child1, i, j] += 1
            cpvals[child2, i, j] += 1
            cpvals[child3, i, j] += 1
            cpvals[child4, i, j] += 1

    # Normalize by 4 (each of the 4 child-allele combos is equally likely):
    cpvals /= 4.0

    # Finally, flatten in 'child-fastest' order, matching IndexToAssignment:
    # (i.e. child dimension varies quickest, then parentOne, then parentTwo).
    val = cpvals[1:, 1:, 1:].flatten(order="F").tolist()

    # Return factor in the same structure as the MATLAB/Octave code
    return {
        "var": [genotypeVarChild, genotypeVarParentOne, genotypeVarParentTwo],
        "card": [m, m, m],
        "val": val,
    }


if __name__ == "__main__":
    factor = genotypeGivenParentsGenotypesFactor(
        numAlleles=2, genotypeVarChild=3, genotypeVarParentOne=1, genotypeVarParentTwo=2
    )
    print("Factor variables:", factor["var"])
    print("Cardinalities:", factor["card"])
    print("vals:", factor["val"])

    # You should see the same 27 entries:
    # [1,0,0,0.5,0.5,0,0,1,0,0.5,0.5,0,0.25,0.5,0.25,0,0.5,0.5,0,1,0,0,0.5,0.5,0,0,1]

    # If desired, you can also wrap this 3D array into a pgmpy TabularCPD:
    cpd_vals = np.array(factor["val"]).reshape(
        (factor["card"][0], factor["card"][1] * factor["card"][2]), order="F"
    )
    cpd = TabularCPD(
        variable="G_child",
        variable_card=factor["card"][0],
        values=cpd_vals,
        evidence=["G_parent1", "G_parent2"],
        evidence_card=[factor["card"][1], factor["card"][2]],
    )
    print("\nEquivalent TabularCPD in pgmpy:\n", cpd)
