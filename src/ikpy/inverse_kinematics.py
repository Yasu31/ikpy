# coding= utf8
import scipy.optimize
import numpy as np
from . import logs


def inverse_kinematic_optimization(chain, target_frame, starting_nodes_angles, regularization_parameter=None, max_iter=None):
    """Computes the inverse kinematic on the specified target with an optimization method

    :param ikpy.chain.Chain chain: The chain used for the Inverse kinematics.
    :param numpy.array target: The desired target.
    :param numpy.array starting_nodes_angles: The initial pose of your chain.
    :param float regularization_parameter: The coefficient of the regularization.
    :param int max_iter: Maximum number of iterations for the optimisation algorithm.
    """
    # Only get the position
    target = target_frame[:3, 3]

    if starting_nodes_angles is None:
        raise ValueError("starting_nodes_angles must be specified")

    # Compute squared distance to target
    def optimize_target(x):
        # y = np.append(starting_nodes_angles[:chain.first_active_joint], x)
        y = chain.active_to_full(x, starting_nodes_angles)
        fk=chain.forward_kinematics(y)

        squared_distance = np.linalg.norm(fk[:3, -1] - target) + 0.025*np.linalg.norm(fk[2,:3]-target_frame[2,:3])
        # the second part is the cost function to keep the z axis of the target frame aligned with the world's z axis. The coefficient was decided very experimentally- change it as much as 0.1 and it won't work... should find better solution.

        return squared_distance

    # If a regularization is selected
    if regularization_parameter is not None:
        def optimize_total(x):
            regularization = np.linalg.norm(x - starting_nodes_angles[chain.first_active_joint:])
            return optimize_target(x) + regularization_parameter * regularization
    else:
        def optimize_total(x):
            return optimize_target(x)

    # Compute bounds
    real_bounds = [link.bounds for link in chain.links]
    # real_bounds = real_bounds[chain.first_active_joint:]
    real_bounds = chain.active_from_full(real_bounds)

    options = {}
    # Manage iterations maximum
    if max_iter is not None:
        options["maxiter"] = max_iter

        # Add Constraints
    cons = ({'type':'eq','fun': lambda x: x[1] + x[2] + x[4] + x[5]},
            {'type':'eq','fun': lambda x: x[3]}
            )

    # Utilisation d'une optimisation SLSQP_**
    # https://forum.poppy-project.org/t/inverse-kinematic-with-pen-holder-poppy-ergo-jr/2601/7
    # res = scipy.optimize.minimize(optimize_total, chain.active_from_full(starting_nodes_angles), method='SLSQP', bounds=real_bounds, options=options, constraints=cons)

    # Utilisation d'une optimisation L-BFGS-B
    res = scipy.optimize.minimize(optimize_total, chain.active_from_full(starting_nodes_angles), method='L-BFGS-B', bounds=real_bounds, options=options)

    logs.manager.info("Inverse kinematic optimisation OK, done in {} iterations".format(res.nit))

    return(chain.active_to_full(res.x, starting_nodes_angles))
    # return(np.append(starting_nodes_angles[:chain.first_active_joint], res.x))
