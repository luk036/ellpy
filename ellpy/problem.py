from .cutting_plane import cutting_plane_dc, Options


class Problem:
    _optim_value = None
    _optim_var = None
    _status = None
    _solver_stats = None
    REGISTERED_SOLVE_METHODS = {}

    def __init__(self, S, oracle, options=Options()):
        """[summary]
        
        Arguments:
            S {[type]} -- [description]
            oracle {[type]} -- [description]
        
        Keyword Arguments:
            options {[type]} -- [description] (default: {Options()})
        """
        self.S = S
        self.oracle = oracle
        self.options = options

    @property
    def optim_value(self):
        """The optimal value from the last time the problem was solved.

        Returns
        -------
        float or None
        """
        return self._optim_value

    @property
    def optim_var(self):
        """The optimal value from the last time the problem was solved.

        Returns
        -------
        x_best or None
        """
        return self._optim_var

    @property
    def status(self):
        """The status from the last time the problem was solved.

        Returns
        -------
        str
        """
        return self._status

    @property
    def solver_stats(self):
        """Returns an object containing additional information returned by the solver.
        """
        return self._solver_stats

    def solve(self, *args, **kwargs):
        """Solves the problem using the specified method.

        Parameters
        ----------
        method : function
            The solve method to use.
        solver : str, optional
            The solver to use.
        verbose : bool, optional
            Overrides the default of hiding solver output.
        solver_specific_opts : dict, optional
            A dict of options that will be passed to the specific solver.
            In general, these options will override any default settings
            imposed by cvxpy.

        Returns
        -------
        float
            The optimal value for the problem, or a string indicating
            why the problem could not be solved.
        """
        # func_name = kwargs.pop("method", None)
        # if func_name is not None:
        #     func = Problem.REGISTERED_SOLVE_METHODS[func_name]
        #     return func(self, *args, **kwargs)
        return self._solve(*args, **kwargs)

    @classmethod
    def register_solve(cls, name, func):
        """Adds a solve method to the Problem class.

        Parameters
        ----------
        name : str
            The keyword for the method.
        func : function
            The function that executes the solve method.
        """
        cls.REGISTERED_SOLVE_METHODS[name] = func

    def _solve(self,
               t
               # solver=None,
               # warm_start=False,
               # verbose=False, **kwargs
               ):
        """Solves a DCP compliant optimization problem.

        Saves the values of primal and dual variables in the variable
        and constraint objects, respectively.

        Parameters
        ----------
        t : Best-so-far value
        solver : str, optional
            The solver to use. Defaults to cutting_plane_dc
        warm_start : bool, optional
            Should the previous solver result be used to warm start?
        verbose : bool, optional
            Overrides the default of hiding solver output.
        kwargs : dict, optional
            A dict of options that will be passed to the specific solver.
            In general, these options will override any default settings
            imposed by ellpy.

        Returns
        -------
        float
            The optimal value for the problem, or a string indicating
            why the problem could not be solved.
        """
        xb, fb, num_iters, feasible, status = cutting_plane_dc(
            self.oracle, self.S, t, self.options)

        if feasible:
            if status == 2:
                self._status = 'optimal'
            else:
                self._status = 'feasible'
            self._optim_value = fb
            self._optim_var = xb
        else:
            if status == 3:  # ???
                self._status = 'infeasible'

        solver_stats = SolverStats('deep-cut')
        solver_stats.num_iters = num_iters
        self._solver_stats = solver_stats

        return self.optim_value

    # def _handle_no_solution(self, status):
    #     """Updates value fields when the problem is infeasible or unbounded.

    #     Parameters
    #     ----------
    #     status: str
    #         The status of the solver.
    #     """
    #     # Set all primal and dual variable values to None.
    #     for var_ in self.variables():
    #         var_.save_value(None)
    #     for constr in self.constraints:
    #         constr.save_value(None)
    #     # Set the problem value.
    #     if status in [s.INFEASIBLE, s.INFEASIBLE_INACCURATE]:
    #         self._value = self.objective.primal_to_result(np.inf)
    #     elif status in [s.UNBOUNDED, s.UNBOUNDED_INACCURATE]:
    #         self._value = self.objective.primal_to_result(-np.inf)


class SolverStats:
    """Reports some of the miscellaneous information that is returned
    by the solver after solving but that is not captured directly by
    the Problem instance.

    Attributes
    ----------
    num_iters : int
        The number of iterations the solver had to go through to find a solution.
    """

    def __init__(self, solver_name):
        """[summary]
        
        Arguments:
            solver_name {[type]} -- [description]
        """
        self.solver_name = solver_name
        self.num_iters = None
