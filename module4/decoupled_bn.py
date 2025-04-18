from typing import List, Union, Dict, Optional
import numpy as np
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.factors.discrete import TabularCPD


def childCopyGivenParentalsFactor(
    numAlleles, geneCopyVarChild, geneCopyVarOne, geneCopyVarTwo
):
    """
    This replicates childCopyGivenParentalsFactor.m. It builds a factor P(ChildAllele | ParentAllele1, ParentAllele2) whose entries encode that the child inherits either the parent's first gene-copy or second gene-copy (maternal/paternal) with probability 0.5 each, except in the case where both of the parent's copies are the same, in which case the child inherits that same allele with probability 1.
    """
    # We will create a 1D array of length numAlleles^3 (in the same order
    # as the original nested for-loops).  Then we pass it to DiscreteFactor.
    values = np.zeros(numAlleles**3)

    # alleleProbsCount will track our position in the 1D array
    alleleProbsCount = 0

    # NOTE: i, j, k are each 0-based here,
    # but this exactly mirrors i=1..numAlleles in the Octave code.
    for i in range(numAlleles):
        for j in range(numAlleles):
            for k in range(numAlleles):
                val = 0.0
                # Compare to the original logic:
                # if (j == k)
                #    if (i == k) val=1
                #    else        val=0.5
                # elif (i == k) val=0.5
                if j == k:
                    if i == k:
                        val = 1.0
                    else:
                        val = 0.5
                elif i == k:
                    val = 0.5

                values[alleleProbsCount] = val
                alleleProbsCount += 1

    # Build a pgmpy factor.  The ordering of 'variables' and 'cardinality'
    # matches the Octave factor.var = [child, parentOne, parentTwo]
    factor = DiscreteFactor(
        variables=[geneCopyVarChild, geneCopyVarOne, geneCopyVarTwo],
        cardinality=[numAlleles, numAlleles, numAlleles],
        values=values,
    )
    return factor


def childCopyGivenFreqsFactor(alleleFreqs, geneCopyVar):
    """
    This replicates childCopyGivenFreqsFactor.m.  It simply makes a factor
    (1D) giving P(GeneCopy) = alleleFreqs.
    """
    numAlleles = len(alleleFreqs)

    factor = DiscreteFactor(
        variables=[geneCopyVar], cardinality=[numAlleles], values=alleleFreqs
    )
    return factor


def generateAlleleGenotypeMappers(numAlleles):
    """
    In the course/Octave code, we have two matrices:

      1) allelesToGenotypes(i, j)
         = the "genotype ID" corresponding to alleles i and j (i <= j).
      2) genotypesToAlleles(k, :)
         = the pair of alleles that form genotype k.

    For simplicity below, we implement 0-based versions,
    but the idea is the same: we enumerate all (i,j) with i <= j.
    We assign them consecutive genotype IDs in lex order.
    """
    # We will store i<=j in ascending lex order.
    # genotypeID -> (i,j)
    genotypesToAlleles = []

    for i in range(numAlleles):
        for j in range(i, numAlleles):
            genotypesToAlleles.append((i, j))

    # Now build the reverse mapping, a numAlleles x numAlleles matrix
    # where (i, j) or (j, i) is mapped to the appropriate genotype ID.
    allelesToGenotypes = np.zeros((numAlleles, numAlleles), dtype=int)
    for gtID, (i, j) in enumerate(genotypesToAlleles):
        allelesToGenotypes[i, j] = gtID
        allelesToGenotypes[j, i] = gtID  # symmetric if i != j

    return allelesToGenotypes, genotypesToAlleles


def phenotypeGivenCopiesFactor(
    alphaList, numAlleles, geneCopyVarOne, geneCopyVarTwo, phenotypeVar
):
    """
    Replicates phenotypeGivenCopiesFactor.m exactly.  We build a factor
    whose entries encode P(Phenotype | Copy1, Copy2).

    alphaList has one entry per genotype, giving P(Phenotype=1 | that genotype).
    The matrix allelesToGenotypes maps (copy1, copy2) => genotypeID,
    so we can look up the correct alpha.
    """
    # Build the genotype mappers
    allelesToGenotypes, genotypesToAlleles = generateAlleleGenotypeMappers(numAlleles)

    # We have 2 phenotypes: 1 => "has trait", 2 => "no trait".
    # In 0-based Python, we can interpret phenotype=0 => "has trait", phenotype=1 => "no trait".
    # But to match the EXACT .val ordering, we will keep the same loops as the Octave code.

    # The factor has variables [phenotypeVar, geneCopyVarOne, geneCopyVarTwo],
    # with cardinalities [2, numAlleles, numAlleles].
    card = [2, numAlleles, numAlleles]
    values = np.zeros(np.prod(card))

    # We fill in values by enumerating all assignments to (phenotype, copy1, copy2).
    # Index i in [0..prod(card)-1]. We'll decode i -> (p, a1, a2).
    # Then if p=1 => prob = alpha(genotype), else => 1 - alpha.
    # NOTE:  In the original code, phenotype=1 => "has trait", phenotype=2 => "no trait".
    # so we do: if p==0 => alphaList(...), else => 1 - alphaList(...).

    # A convenient way is to do triple nested loops in the same order that
    # IndexToAssignment would produce:
    idx = 0
    for p in range(2):  # phenotype: 0 => "has trait", 1 => "no trait"
        for a1 in range(numAlleles):
            for a2 in range(numAlleles):
                # figure out genotype
                genotypeID = allelesToGenotypes[a1, a2]
                alpha = alphaList[genotypeID]  # alphaList is 0-based
                if p == 0:
                    # p=0 => "has trait"
                    values[idx] = alpha
                else:
                    values[idx] = 1.0 - alpha
                idx += 1

    factor = DiscreteFactor(
        variables=[phenotypeVar, geneCopyVarOne, geneCopyVarTwo],
        cardinality=card,
        values=values,
    )
    return factor


FactorType = Union[DiscreteFactor, TabularCPD, Dict]


def constructDecoupledGeneticNetwork(pedigree, alleleFreqs, alphaList):
    """
    Replicates constructDecoupledGeneticNetwork.m in Python/pgmpy.

    Inputs
    ------
    pedigree : a dict-like object with fields:
        parents : (N x 2) numpy array of integer indices
                  if parents[i,0] == 0, person i has no father
                  if parents[i,1] == 0, person i has no mother
        names   : list of strings of length N
    alleleFreqs : 1D list or array of length numAlleles
                  The population frequencies of each allele
    alphaList   : 1D list or array of length m
                  alphaList[g] = P(has-trait | genotype g)

    Outputs
    -------
    factorList : list of DiscreteFactor objects of length 3*N,
                 storing the CPDs for each person's:
                  1) gene-copy-1 factor
                  2) gene-copy-2 factor
                  3) phenotype factor

    Variable numbering:
      - Person i (1-based) has:
          gene-copy-1 variable = i
          gene-copy-2 variable = i + N
          phenotype variable   = i + 2N
      - Hence factorList[i-1]   is for person i's gene-copy-1,
        factorList[N + i - 1]   is for person i's gene-copy-2,
        factorList[2N + i - 1]  is for person i's phenotype.
    """

    # Number of people
    numPeople = len(pedigree["names"])
    numAlleles = len(alleleFreqs)

    # Initialize a Python list of length 3 * numPeople
    factorList: List[Optional[FactorType]] = [None] * (3 * numPeople)

    # Bookkeeping array to keep track of whether we've assigned
    # the gene-copy factors for each person (1 => not assigned, 0 => assigned)
    bookkeeping = [1] * numPeople

    # First pass:  Founders (those with father=0 => mother=0)
    # get childCopyGivenFreqsFactor()
    for i in range(1, numPeople + 1):  # i=1..numPeople
        father = pedigree["parents"][i - 1, 0]
        # If father==0 => mother==0 as well, so no parents
        if father == 0:
            # gene copy 1 factor
            factorList[i - 1] = childCopyGivenFreqsFactor(alleleFreqs, i)
            # gene copy 2 factor
            factorList[numPeople + i - 1] = childCopyGivenFreqsFactor(
                alleleFreqs, i + numPeople
            )
            bookkeeping[i - 1] = 0

    # Second pass:  Non-founders get childCopyGivenParentalsFactor
    # but only after their parents' gene-copy factors have been assigned
    while sum(bookkeeping) != 0:
        for i in range(1, numPeople + 1):
            if bookkeeping[i - 1] == 1:
                father = pedigree["parents"][i - 1, 0]
                mother = pedigree["parents"][i - 1, 1]
                # If father>0 => father is a valid index => check if father done
                if father > 0:
                    father_done = bookkeeping[father - 1] == 0
                else:
                    # father=0 means we would have assigned in the first pass
                    father_done = True

                if mother > 0:
                    mother_done = bookkeeping[mother - 1] == 0
                else:
                    mother_done = True

                # Only assign this person's factors if both parents are done
                if father_done and mother_done:
                    # gene copy 1 factor (child inherits from father)
                    if father != 0:
                        factorList[i - 1] = childCopyGivenParentalsFactor(
                            numAlleles,
                            i,  # child's gene-copy-1 var
                            father,  # father's gene-copy-1 variable
                            father + numPeople,  # father's gene-copy-2 variable
                        )
                    # gene copy 2 factor (child inherits from mother)
                    if mother != 0:
                        factorList[numPeople + i - 1] = childCopyGivenParentalsFactor(
                            numAlleles,
                            i + numPeople,  # child's gene-copy-2 var
                            mother,  # mother's gene-copy-1 variable
                            mother + numPeople,  # mother's gene-copy-2 variable
                        )
                    bookkeeping[i - 1] = 0

    # Third pass: phenotype factors
    # P(Phenotype_i | GeneCopy1_i, GeneCopy2_i)
    for i in range(1, numPeople + 1):
        factorList[2 * numPeople + i - 1] = phenotypeGivenCopiesFactor(
            alphaList,  # alphaList for genotype->trait probabilities
            numAlleles,
            i,  # gene-copy-1 var
            i + numPeople,  # gene-copy-2 var
            i + 2 * numPeople,  # phenotype var
        )

    return factorList


if __name__ == "__main__":
    # # Suppose we have alphaList for 3-allele system:
    # alphaListThree = np.array([0.8, 0.6, 0.1, 0.5, 0.05, 0.01])
    # # That corresponds to genotypes:
    # #  (1,1)->0.8,  (1,2)->0.6,  (1,3)->0.1,  (2,2)->0.5,  (2,3)->0.05, (3,3)->0.01

    # numAllelesThree = 3
    # genotypeVarMotherCopy = "MotherCopy"
    # genotypeVarFatherCopy = "FatherCopy"
    # phenotypeVar = "Phenotype"

    # # Build the factor in Python:
    # phenotypeFactor = phenotypeGivenCopiesFactor(
    #     alphaListThree,
    #     numAllelesThree,
    #     genotypeVarMotherCopy,
    #     genotypeVarFatherCopy,
    #     phenotypeVar,
    # )

    # # Print out the .values flattened, to compare with the Octave struct .val:
    # # In Octave, the reference solution had .val of length 18:
    # print("Phenotype factor scope =", phenotypeFactor.scope())
    # print("Phenotype factor cardinalities =", phenotypeFactor.cardinality)
    # print("Phenotype factor values (flattened) =")
    # print(phenotypeFactor.values.flatten())

    pedigree = {
        "parents": np.array([[0, 0], [1, 3], [0, 0]]),  # father,mother
        "names": ["Ira", "James", "Robin"],
    }
    alleleFreqsThree = np.array([0.1, 0.7, 0.2])
    alphaListThree = np.array([0.8, 0.6, 0.1, 0.5, 0.05, 0.01])

    # Build the decoupled genetic network:
    factorList = constructDecoupledGeneticNetwork(
        pedigree, alleleFreqsThree, alphaListThree
    )

    # We now have a list of 9 factors (for this 3-person pedigree).
    # factorList[0] => Ira's gene-copy-1 factor
    # factorList[1] => James's gene-copy-1 factor
    # ...
    # factorList[6] => Ira's phenotype factor, etc.

    # You can compare .values and .scope() to the Octave sampleFactorListDecoupled
    for idx, f in enumerate(factorList):
        print(
            f"Factor #{idx+1} has scope {f.scope()} with cardinalities {f.cardinality}"
        )
        print("Values (flattened) =", f.values.flatten(), "\n")
