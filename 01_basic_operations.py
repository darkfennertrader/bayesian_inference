import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD, DiscreteFactor


cpd = TabularCPD(
    variable="grade",
    variable_card=3,
    values=[
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
    ],
    evidence=["diff", "intel"],
    evidence_card=[2, 3],
    state_names={
        "diff": ["easy", "hard"],
        "intel": ["low", "mid", "high"],
        "grade": ["A", "B", "C"],
    },
)
# print(cpd)

############################################################################3
# (1) JOINT DISTRIBUTION FOR INDEPENDENT VARIABLES

# Define the first CPD
cpd_A = TabularCPD(variable="A", variable_card=2, values=[[0.7], [0.3]])

# Define the second CPD
cpd_B = TabularCPD(variable="B", variable_card=2, values=[[0.4], [0.6]])

# Multiply the CPDs
product_cpd = cpd_A.to_factor() * cpd_B.to_factor()

# The result is a DiscreteFactor, you can convert it back to a CPD if needed
# print(product_cpd)

#########################################################################
# (2) Joint Probability of X1-> X2 Bayesian Model
model = BayesianNetwork([("X1", "X2")])

# P(X1)
cpd_X1 = TabularCPD(variable="X1", variable_card=2, values=[[0.6], [0.4]])

# P(X2 | X1)
cpd_X2_given_X1 = TabularCPD(
    variable="X2",
    variable_card=2,
    values=[[0.7, 0.2], [0.3, 0.8]],
    evidence=["X1"],
    evidence_card=[2],
)

factor_X1 = DiscreteFactor(["X1"], [2], cpd_X1.values.flatten())
print("\nFactor X1:")
print(factor_X1)
factor_X2_given_X1 = DiscreteFactor(
    ["X2", "X1"], [2, 2], cpd_X2_given_X1.values.flatten()
)

joint_factor = factor_X1.product(factor_X2_given_X1, inplace=False)
joint_cpd = TabularCPD(
    variable="X2",
    variable_card=2,
    values=joint_factor.values.reshape(2, 2).tolist(),
    evidence=["X1"],
    evidence_card=[2],
)


print("\nJoint CPD of X1 and X2:")
print(joint_cpd)


# Convert CPDs to factors
factor_X1 = cpd_X1.to_factor()
factor_X2_given_X1 = cpd_X2_given_X1.to_factor()

# Multiply the factors
product_factor = factor_X1 * factor_X2_given_X1

# Print the resulting factor
print("\n Product of factors:")
print(product_factor)
print("\n Array of product of factors:")
print(product_factor.values)


###################################################################
# FACTOR MARGINALIZATION

# Define a CPD for a node 'A' with 2 states and one parent 'B' with 2 states
# Define the CPD
cpd_a_given_b = TabularCPD(
    variable="A",
    variable_card=2,
    values=[[0.8, 0.3], [0.2, 0.7]],
    evidence=["B"],
    evidence_card=[2],
)

# Convert the CPD to a factor to perform marginalization
factor_a_given_b = cpd_a_given_b.to_factor()

# Define the marginal distribution of B (e.g., P(B)) called also EVIDENCE
marginal_b = TabularCPD(variable="B", variable_card=2, values=[[0.2], [0.8]])

# Convert marginal_b to a factor
factor_b = marginal_b.to_factor()

# Multiply the factors to get the joint distribution P(A, B)
joint_factor = factor_a_given_b * factor_b

# Marginalize over B to get P(A)
marginal_a = joint_factor.marginalize(variables=["B"], inplace=False)

print("\nMarginalizing out B:")
print(marginal_a)


###################################################################
# OBSERVE EVIDENCE


def observe_evidence_vectorized(factor, evidence):
    # Create a boolean mask initially set to True for all entries
    mask = np.ones(factor.values.shape, dtype=bool)

    # For each piece of evidence, update the mask
    for var, value in evidence.items():
        var_index = factor.variables.index(var)
        # Create an index array that selects only the desired evidence value
        slices = [slice(None)] * len(factor.cardinality)
        slices[var_index] = value

        # Update the mask to zero out inconsistent entries
        mask &= np.indices(factor.cardinality)[var_index] == value

    # Apply the mask to zero out inconsistent entries in factor values
    factor.values[~mask] = 0


# Define a simple factor with variables X1 and X2
variables = ["X1", "X2"]
cardinalities = [2, 2]
values = np.array([0.1, 0.9, 0.8, 0.2]).reshape(cardinalities)

factor = DiscreteFactor(variables, cardinalities, values)

print("\nFACTOR:")
print(factor)

# Define evidence: X1 = 0
evidence = {"X1": 0}

# Observe the evidence using the vectorized function
observe_evidence_vectorized(factor, evidence)

print("\nModified Factor Values:")
print(factor.values)
