# -*- coding: utf-8 -*-
"""Tools to compute KnowledgeGradient and optimize the next best point(s) to sample using KG through C++ calls.

This file contains a class to compute Knowledge Gradient + derivatives and a functions to solve the q,p-KG optimization problem.
The :class:`moe.optimal_learning.python.cpp_wrappers.knowledge_gradient.KnowledgeGradient`
The optimization functions are convenient wrappers around the matching C++ calls.

See gpp_knowledge_gradient_optimization.hpp/cpp for further details on knowledge gradient.

"""
import numpy

import moe.build.GPP as C_GP
from moe.optimal_learning.python.constant import DEFAULT_EXPECTED_IMPROVEMENT_MC_ITERATIONS, DEFAULT_MAX_NUM_THREADS
import moe.optimal_learning.python.cpp_wrappers.cpp_utils as cpp_utils
from moe.optimal_learning.python.interfaces.optimization_interface import OptimizableInterface


def posterior_mean_optimization(
        ps_optimizer,
        initial_guess,
        randomness=None,
        max_num_threads=DEFAULT_MAX_NUM_THREADS,
        status=None,
):
    """Solve the q,p-KG problem, returning the optimal set of q points to sample CONCURRENTLY in future experiments.

    .. NOTE:: The following comments are copied from gpp_math.hpp, ComputeOptimalPointsToSample().
      These comments are copied into
      :func:`moe.optimal_learning.python.python_version.expected_improvement.multistart_expected_improvement_optimization`

    This is the primary entry-point for EI optimization in the optimal_learning library. It offers our best shot at
    improving robustness by combining higher accuracy methods like gradient descent with fail-safes like random/grid search.

    Returns the optimal set of q points to sample CONCURRENTLY by solving the q,p-EI problem.  That is, we may want to run 4
    experiments at the same time and maximize the EI across all 4 experiments at once while knowing of 2 ongoing experiments
    (4,2-EI). This function handles this use case. Evaluation of q,p-EI (and its gradient) for q > 1 or p > 1 is expensive
    (requires monte-carlo iteration), so this method is usually very expensive.

    Compared to ComputeHeuristicPointsToSample() (``gpp_heuristic_expected_improvement_optimization.hpp``), this function
    makes no external assumptions about the underlying objective function. Instead, it utilizes a feature of the
    GaussianProcess that allows the GP to account for ongoing/incomplete experiments.

    If ``num_to_sample = 1``, this is the same as ComputeOptimalPointsToSampleWithRandomStarts().

    The option of using GPU to compute general q,p-EI via MC simulation is also available. To enable it, make sure you have
    installed GPU components of MOE, otherwise, it will throw Runtime excpetion.

    :param kg_optimizer: object that optimizes (e.g., gradient descent, newton) EI over a domain
    :type kg_optimizer: cpp_wrappers.optimization.*Optimizer object
    :param num_multistarts: number of times to multistart ``ei_optimizer`` (UNUSED, data is in ei_optimizer.optimizer_parameters)
    :type num_multistarts: int > 0
    :param num_to_sample: how many simultaneous experiments you would like to run (i.e., the q in q,p-EI)
    :type num_to_sample: int >= 1
    :param use_gpu: set to True if user wants to use GPU for MC simulation
    :type use_gpu: bool
    :param which_gpu: GPU device ID
    :type which_gpu: int >= 0
    :param randomness: RNGs used by C++ to generate initial guesses and as the source of normal random numbers when monte-carlo is used
    :type randomness: RandomnessSourceContainer (C++ object; e.g., from C_GP.RandomnessSourceContainer())
    :param max_num_threads: maximum number of threads to use, >= 1
    :type max_num_threads: int > 0
    :param status: (output) status messages from C++ (e.g., reporting on optimizer success, etc.)
    :type status: dict
    :return: point(s) that maximize the knowledge gradient (solving the q,p-KG problem)
    :rtype: array of float64 with shape (num_to_sample, ei_optimizer.objective_function.dim)

    """
    # Create enough randomness sources if none are specified.
    if randomness is None:
        randomness = C_GP.RandomnessSourceContainer(max_num_threads)
        # Set seeds based on less repeatable factors (e.g,. time)
        randomness.SetRandomizedUniformGeneratorSeed(0)
        randomness.SetRandomizedNormalRNGSeed(0)

    # status must be an initialized dict for the call to C++.
    if status is None:
        status = {}

    best_points_to_sample = C_GP.posterior_mean_optimization(
        ps_optimizer.objective_function._gaussian_process._gaussian_process,
        ps_optimizer.objective_function._num_fidelity,
        ps_optimizer.optimizer_parameters,
        [float(x) for x in cpp_utils.cppify(ps_optimizer.domain.domain_bounds)],
        cpp_utils.cppify(initial_guess),
        status,
    )

    # reform output to be a list of dim-dimensional points, dim = len(self.domain)
    return numpy.array(best_points_to_sample)

class PosteriorMean(OptimizableInterface):
    def __init__(
            self,
            gaussian_process,
            num_fidelity=0,
            points_to_sample=None,
            randomness=None,
    ):
        self._gaussian_process = gaussian_process
        self._num_fidelity = num_fidelity

        if points_to_sample is None:
            self._points_to_sample = numpy.zeros((1, self._gaussian_process.dim))

        if randomness is None:
            self._randomness = C_GP.RandomnessSourceContainer(1)  # create randomness for only 1 thread
            # Set seed based on less repeatable factors (e.g,. time)
            self._randomness.SetRandomizedUniformGeneratorSeed(0)
            self._randomness.SetRandomizedNormalRNGSeed(0)
        else:
            self._randomness = randomness

        self.objective_type = None  # Not used for KG, but the field is expected in C++

    @property
    def dim(self):
        """Return the number of spatial dimensions."""
        return self._gaussian_process.dim

    @property
    def problem_size(self):
        """Return the number of independent parameters to optimize."""
        return self.dim-self._num_fidelity

    def get_current_point(self):
        """Get the current_point (array of float64 with shape (problem_size)) at which this object is evaluating the objective function, ``f(x)``."""
        return numpy.copy(self._points_to_sample)

    def set_current_point(self, points_to_sample):
        """Set current_point to the specified point; ordering must match.
        :param points_to_sample: current_point at which to evaluate the objective function, ``f(x)``
        :type points_to_sample: array of float64 with shape (problem_size)
        """
        self._points_to_sample = numpy.copy(numpy.atleast_2d(points_to_sample))

    current_point = property(get_current_point, set_current_point)

    def compute_posterior_mean(self, force_monte_carlo=False):
        r"""Compute the knowledge gradient at ``points_to_sample``, with ``points_being_sampled`` concurrent points being sampled.

        .. Note:: These comments were copied from
          :meth:`moe.optimal_learning.python.interfaces.expected_improvement_interface.ExpectedImprovementInterface.compute_expected_improvement`

        ``points_to_sample`` is the "q" and ``points_being_sampled`` is the "p" in q,p-EI.

        Computes the knowledge gradient ``EI(Xs) = E_n[[f^*_n(X) - min(f(Xs_1),...,f(Xs_m))]^+]``, where ``Xs``
        are potential points to sample (union of ``points_to_sample`` and ``points_being_sampled``) and ``X`` are
        already sampled points.  The ``^+`` indicates that the expression in the expectation evaluates to 0 if it
        is negative.  ``f^*(X)`` is the MINIMUM over all known function evaluations (``points_sampled_value``),
        whereas ``f(Xs)`` are *GP-predicted* function evaluations.

        In words, we are computing the knowledge gradient (over the current ``best_so_far``, best known
        objective function value) that would result from sampling (aka running new experiments) at
        ``points_to_sample`` with ``points_being_sampled`` concurrent/ongoing experiments.

        In general, the EI expression is complex and difficult to evaluate; hence we use Monte-Carlo simulation to approximate it.
        When faster (e.g., analytic) techniques are available, we will prefer them.

        The idea of the MC approach is to repeatedly sample at the union of ``points_to_sample`` and
        ``points_being_sampled``. This is analogous to gaussian_process_interface.sample_point_from_gp,
        but we sample ``num_union`` points at once:
        ``y = \mu + Lw``
        where ``\mu`` is the GP-mean, ``L`` is the ``chol_factor(GP-variance)`` and ``w`` is a vector
        of ``num_union`` draws from N(0, 1). Then:
        ``improvement_per_step = max(max(best_so_far - y), 0.0)``
        Observe that the inner ``max`` means only the smallest component of ``y`` contributes in each iteration.
        We compute the improvement over many random draws and average.

        :param force_monte_carlo: whether to force monte carlo evaluation (vs using fast/accurate analytic eval when possible)
        :type force_monte_carlo: boolean
        :return: the knowledge gradient from sampling ``points_to_sample`` with ``points_being_sampled`` concurrent experiments
        :rtype: float64

        """
        return C_GP.compute_posterior_mean(
            self._gaussian_process._gaussian_process,
            self._num_fidelity,
            cpp_utils.cppify(self._points_to_sample),
        )

    compute_objective_function = compute_posterior_mean

    def compute_grad_posterior_mean(self, force_monte_carlo=False):
        r"""Compute the gradient of knowledge gradient at ``points_to_sample`` wrt ``points_to_sample``, with ``points_being_sampled`` concurrent samples.

        .. Note:: These comments were copied from
          :meth:`moe.optimal_learning.python.interfaces.expected_improvement_interface.ExpectedImprovementInterface.compute_grad_expected_improvement`

        ``points_to_sample`` is the "q" and ``points_being_sampled`` is the "p" in q,p-EI.

        In general, the expressions for gradients of EI are complex and difficult to evaluate; hence we use
        Monte-Carlo simulation to approximate it. When faster (e.g., analytic) techniques are available, we will prefer them.

        The MC computation of grad EI is similar to the computation of EI (decsribed in
        compute_expected_improvement). We differentiate ``y = \mu + Lw`` wrt ``points_to_sample``;
        only terms from the gradient of ``\mu`` and ``L`` contribute. In EI, we computed:
        ``improvement_per_step = max(max(best_so_far - y), 0.0)``
        and noted that only the smallest component of ``y`` may contribute (if it is > 0.0).
        Call this index ``winner``. Thus in computing grad EI, we only add gradient terms
        that are attributable to the ``winner``-th component of ``y``.

        :param force_monte_carlo: whether to force monte carlo evaluation (vs using fast/accurate analytic eval when possible)
        :type force_monte_carlo: boolean
        :return: gradient of EI, ``\pderiv{EI(Xq \cup Xp)}{Xq_{i,d}}`` where ``Xq`` is ``points_to_sample``
          and ``Xp`` is ``points_being_sampled`` (grad EI from sampling ``points_to_sample`` with
          ``points_being_sampled`` concurrent experiments wrt each dimension of the points in ``points_to_sample``)
        :rtype: array of float64 with shape (num_to_sample, dim)

        """
        grad_kg = C_GP.compute_grad_posterior_mean(
            self._gaussian_process._gaussian_process,
            self._num_fidelity,
            cpp_utils.cppify(self._points_to_sample),
        )
        return cpp_utils.uncppify(grad_kg, (1, self.dim-self._num_fidelity))

    compute_grad_objective_function = compute_grad_posterior_mean

    def compute_hessian_objective_function(self, **kwargs):
        """We do not currently support computation of the (spatial) hessian of knowledge gradient."""
        raise NotImplementedError('Currently we cannot compute the hessian of knowledge gradient.')

