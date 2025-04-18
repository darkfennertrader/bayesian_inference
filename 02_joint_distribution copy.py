import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import DiscreteFactor, TabularCPD


class JointDistributionBuilder:
    """
    A helper class to construct the joint distribution (as a DiscreteFactor)
    from a BayesianNetwork model and its CPDs.
    """

    def __init__(self, bayesian_model: BayesianNetwork, tabular_cpds: list):
        """
        Parameters
        ----------
        model : BayesianNetwork
            An instance of a pgmpy BayesianNetwork.
        cpds : list
            A list of CPDs (TabularCPD objects) corresponding to 'model'.
        """
        self.model = bayesian_model  # Activates the setter
        self.cpds = tabular_cpds  # Activates the setter
        # Build the joint factor (will use validated model & CPDs)
        self.joint_factor = self.build_joint_distribution()

        @property
        def model(self) -> BayesianNetwork:
            """
            Getter for the BayesianNetwork.
            """
            return self._model

        @model.setter
        def model(self, value: BayesianNetwork):
            """
            Setter that validates the model before assigning it internally.
            """
            if not isinstance(value, BayesianNetwork):
                raise TypeError("Provided model is not a BayesianNetwork instance.")
            self._model = value

        @property
        def cpds(self) -> list:
            """
            Getter for the list of CPDs.
            """
            return self._cpds

        @cpds.setter
        def cpds(self, value: list):
            """
            Setter that validates the CPDs before assigning them internally.
            Includes checks for:
            - Each CPD is an instance of TabularCPD.
            - Each variable in the model has exactly one CPD.
            - Parent sets in the CPDs match the DAG structure in the model.
            """
            if not isinstance(value, list):
                raise TypeError(
                    "'cpds' must be provided as a list of CPDs (TabularCPD)."
                )

            # Ensure the model is already set, if not, disallow setting CPDs
            if not hasattr(self, "_model"):
                raise AttributeError("Model must be set before CPDs.")

            # Validate each CPD
            for cpd in value:
                if not isinstance(cpd, TabularCPD):
                    raise TypeError("All CPDs must be instances of 'TabularCPD'.")
                if cpd.variable not in self.model.nodes():
                    raise ValueError(
                        f"CPD variable '{cpd.variable}' not found in the model."
                    )
                # Check parent consistency
                model_parents = set(self.model.get_parents(cpd.variable))
                cpd_parents = set(cpd.variables[1:])
                if model_parents != cpd_parents:
                    raise ValueError(
                        f"Parents for variable '{cpd.variable}' in CPD do not match "
                        f"the parents in the DAG: {model_parents} != {cpd_parents}."
                    )

            # Check there is exactly one CPD for every variable in the model
            model_vars = set(self.model.nodes())
            cpd_vars = set(cpd.variable for cpd in value)
            if model_vars != cpd_vars:
                raise ValueError(
                    "Mismatch between model variables and the CPDs provided. "
                    f"Model variables: {model_vars}, CPD variables: {cpd_vars}"
                )

            self._cpds = value

    def build_joint_distribution(self) -> DiscreteFactor:
        """
        Converts each CPD to a DiscreteFactor and multiplies them to get the joint.
        Returns
        -------
        DiscreteFactor
            The joint distribution factor of all CPDs combined.
        """
        # Convert each CPD to a factor
        factors = [cpd.to_factor() for cpd in self.cpds]

        # Multiply all the factors together
        joint_factor = factors[0].copy()
        for factor in factors[1:]:
            joint_factor.product(factor, inplace=True)

        return joint_factor

    def get_joint_value(self, **variable_states) -> float:
        """
        Get the probability value from the joint distribution for given variable states.

        Parameters
        ----------
        variable_states : keyword arguments
            Example: X1=0, X2=1, etc. The states can be integer indices or actual state names.

        Returns
        -------
        float
            The joint probability associated with the specified states.
        """
        return self.joint_factor.get_value(**variable_states)

    def get_assignment(self, index: int):
        """
        Returns the assignment (as a list of (variable, state) pairs)
        for a specific index in the joint factor.

        Parameters
        ----------
        index : int
            Index into the factor's values array.

        Returns
        -------
        list of tuples
            A list of (variable_name, variable_state) pairs.
        """
        # DiscreteFactor.assignment expects a list of indices
        return self.joint_factor.assignment([index])[0]

    def scope(self):
        """
        Returns the list of variables that this joint distribution covers.
        """
        return self.joint_factor.scope()


if __name__ == "__main__":
    # Define a simple Bayesian Network
    model = BayesianNetwork([("X1", "X2")])

    # Define CPDs
    cpd_X1 = TabularCPD(variable="X1", variable_card=2, values=[[0.6], [0.4]])

    cpd_X2_given_X1 = TabularCPD(
        variable="X2",
        variable_card=2,
        values=[[0.7, 0.2], [0.3, 0.8]],
        evidence=["X1"],
        evidence_card=[2],
    )

    # Instantiate JointDistributionBuilder
    builder = JointDistributionBuilder(
        bayesian_model=model, tabular_cpds=[cpd_X1, cpd_X2_given_X1]
    )

    # Print scope of the joint distribution
    print("Scope of the joint distribution:", builder.scope())

    # Get a specific joint probability, e.g., P(X1=0, X2=1)
    print("P(X1=0, X2=1) =", builder.get_joint_value(X1=0, X2=1))

    # Show an assignment example by index into the factor's values.
    # For a factor with two variables [X1, X2] each of cardinality 2,
    # there are 4 assignments (index 0..3).
    for idx in range(4):
        print(
            f"Index {idx} -> assignment: {builder.get_assignment(idx)}, "
            f"value: {builder.joint_factor.values.flat[idx]:.4f}"
        )

    print("\nJoint Distribution:")
    joint_distr = builder.build_joint_distribution()
    print(joint_distr)
