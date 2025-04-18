import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
credit_net = nx.DiGraph()

# Add nodes
nodes = [
    "Income",
    "Age",
    "Ratio_of_Debts_To_Income",
    "Payment_History",
    "Assets",
    "Future_Income",
    "Reliability",
    "Credit_Worthiness",
]
credit_net.add_nodes_from(nodes)

# Add edges
edges = [
    ("Income", "Assets"),
    ("Income", "Future_Income"),
    ("Assets", "Future_Income"),
    ("Age", "Payment_History"),
    ("Ratio_of_Debts_To_Income", "Payment_History"),
    ("Age", "Reliability"),
    ("Payment_History", "Reliability"),
    ("Future_Income", "Credit_Worthiness"),
    ("Ratio_of_Debts_To_Income", "Credit_Worthiness"),
    ("Reliability", "Credit_Worthiness"),
]
credit_net.add_edges_from(edges)

# Draw the graph
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(credit_net)  # positions for all nodes
nx.draw(
    credit_net,
    pos,
    with_labels=True,
    node_size=3000,
    node_color="lightgreen",
    font_size=10,
    font_weight="bold",
    arrowsize=20,
)
plt.title("Bayesian Network Visualization")
plt.savefig("credit_network.png", format="PNG")
plt.show()
