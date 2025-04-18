import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

credit_net = BayesianNetwork()

credit_net.add_node("Income")
credit_net.add_node("Age")
credit_net.add_node("Ratio_of_Debts_To_Income")
credit_net.add_node("Payment_History")
credit_net.add_node("Assets")
credit_net.add_node("Future_Income")
credit_net.add_node("Reliability")
credit_net.add_node("Credit_Worthiness")

credit_net.add_edge("Income", "Assets")
credit_net.add_edge("Income", "Future_Income")
credit_net.add_edge("Assets", "Future_Income")
credit_net.add_edge("Age", "Payment_History")
credit_net.add_edge("Ratio_of_Debts_To_Income", "Payment_History")
credit_net.add_edge("Age", "Reliability")
credit_net.add_edge("Payment_History", "Reliability")
credit_net.add_edge("Future_Income", "Credit_Worthiness")
credit_net.add_edge("Ratio_of_Debts_To_Income", "Credit_Worthiness")
credit_net.add_edge("Reliability", "Credit_Worthiness")

# CPDs definition


cpd_age = TabularCPD(
    variable="Age",
    variable_card=3,
    values=[[0.2], [0.5], [0.3]],
    state_names={"Age": ["16_21", "22_64", "over65"]},
)


cpd_income = TabularCPD(
    variable="Income",
    variable_card=3,
    values=[[0.1], [0.6], [0.3]],
    state_names={"Income": ["high", "medium", "low"]},
)

cpd_ratio_debts = TabularCPD(
    variable="Ratio_of_Debts_To_Income",
    variable_card=2,
    values=[[0.4], [0.6]],
    state_names={"Ratio_of_Debts_To_Income": ["low", "high"]},
)
cpd_assets = TabularCPD(
    variable="Assets",  # The child variable
    variable_card=3,  # Number of states for Assets
    values=[
        [
            0.55,
            0.5,
            0.4,
        ],  # Probabilities for Assets = 'high' given Income = ['high', 'medium', 'low']
        [
            0.3,
            0.25,
            0.2,
        ],  # Probabilities for Assets = 'medium' given Income = ['high', 'medium', 'low']
        [
            0.15,
            0.25,
            0.4,
        ],  # Probabilities for Assets = 'low' given Income = ['high', 'medium', 'low']
    ],
    evidence=["Income"],  # The parent variable
    evidence_card=[3],  # Number of states for Income
    state_names={
        "Assets": ["high", "medium", "low"],
        "Income": ["high", "medium", "low"],
    },
)

# Define the CPD with state names
cpd_future_income = TabularCPD(
    variable="Future_Income",
    variable_card=2,
    values=[
        [
            0.5,
            0.45,
            0.42,
            0.48,
            0.43,
            0.41,
            0.46,
            0.41,
            0.39,
        ],
        [0.5, 0.55, 0.57, 0.52, 0.57, 0.59, 0.54, 0.59, 0.61],
    ],
    evidence=["Income", "Assets"],
    evidence_card=[3, 3],
    state_names={
        "Future_Income": ["promising", "not promising"],
        "Income": ["high", "medium", "low"],
        "Assets": ["high", "medium", "low"],
    },
)


cpd_payment_history = TabularCPD(
    variable="Payment_History",
    variable_card=3,
    values=[
        [0.3, 0.3, 0.5, 0.5, 0.7, 0.6],  # Payment_History = "excellent"
        [0.5, 0.3, 0.3, 0.2, 0.1, 0.1],  # Payment_History = "acceptable"
        [0.2, 0.4, 0.2, 0.3, 0.2, 0.3],  # Payment_History = "unacceptable"
    ],
    evidence=["Age", "Ratio_of_Debts_To_Income"],
    evidence_card=[3, 2],
    state_names={
        "Payment_History": ["excellent", "acceptable", "unacceptable"],
        "Age": ["16_21", "22_64", "over65"],
        "Ratio_of_Debts_To_Income": ["low", "high"],
    },
)
cpd_reliability = TabularCPD(
    variable="Reliability",
    variable_card=2,
    values=[
        [0.7, 0.8, 0.9, 0.6, 0.7, 0.8, 0.55, 0.6, 0.7],
        [0.3, 0.2, 0.1, 0.4, 0.3, 0.2, 0.45, 0.4, 0.3],
    ],
    evidence=["Payment_History", "Age"],
    evidence_card=[3, 3],
    state_names={
        "Reliability": ["reliable", "unreliable"],
        "Payment_History": ["excellent", "acceptable", "unacceptable"],
        "Age": ["16_21", "22_64", "over65"],
    },
)
cpd_credit_worthiness = TabularCPD(
    variable="Credit_Worthiness",
    variable_card=2,
    values=[
        [0.7, 0.6, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35],
        [0.3, 0.4, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
    ],
    evidence=["Reliability", "Ratio_of_Debts_To_Income", "Future_Income"],
    evidence_card=[2, 2, 2],
    state_names={
        "Credit_Worthiness": ["positive", "negative"],
        "Reliability": ["reliable", "unreliable"],
        "Ratio_of_Debts_To_Income": ["low", "high"],
        "Future_Income": ["promising", "not promising"],
    },
)

# Add CPDs to the network
credit_net.add_cpds(
    cpd_age,
    cpd_income,
    cpd_ratio_debts,
    cpd_assets,
    cpd_future_income,
    cpd_payment_history,
    cpd_reliability,
    cpd_credit_worthiness,
)

# Check if the model is valid
assert credit_net.check_model()
