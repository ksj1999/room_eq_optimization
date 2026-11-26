"""Matched L-BFGS-B and projected damped generalized Gauss–Newton fits."""

import numpy as np
from scipy.optimize import OptimizeResult, minimize


def projected_gradient_norm(x, gradient, lower, upper):
    return float(np.linalg.norm(x - np.clip(x - gradient, lower, upper), ord=np.inf))


def lbfgsb(objective, initial, lower, upper, *, budget, tolerance):
    history = []
    evaluations = 0
    gradient = None

    def evaluate(x):
        nonlocal evaluations, gradient
        evaluations += 1
        value, gradient = objective.evaluate(x)
        if evaluations == 1:
            record(x, value)
        return value, gradient

    def record(x, value):
        history.append({
            "evaluation": evaluations, "objective": float(value),
            "projected_gradient": projected_gradient_norm(x, gradient, lower, upper),
        })

    def callback(intermediate_result):
        record(intermediate_result.x, intermediate_result.fun)

    result = minimize(
        evaluate, initial, jac=True, method="L-BFGS-B",
        bounds=list(zip(lower, upper)), callback=callback,
        options={"maxfun": budget, "maxiter": budget, "gtol": tolerance, "ftol": 1e-13},
    )
    result.history = history
    return result


def generalized_gauss_newton(objective, initial, lower, upper, *, budget, tolerance):
    """Fit with projected damped steps and a scaled gradient fallback."""
    x = np.clip(initial, lower, upper)
    value, gradient, hessian = objective.evaluate(x, curvature=True)
    evaluations = hessian_evaluations = 1
    damping = 1e-3
    history = []
    status = "evaluation budget reached"
    step_kind = "initial"
    while True:
        norm = projected_gradient_norm(x, gradient, lower, upper)
        history.append({
            "evaluation": evaluations, "objective": value,
            "projected_gradient": norm, "damping": damping, "step": step_kind,
        })
        if norm <= tolerance:
            status = "projected gradient tolerance reached"
            break
        if evaluations >= budget:
            break
        diagonal = np.maximum(hessian.diagonal(), 1e-3)
        free = ~(((x == lower) & (gradient > 0)) | ((x == upper) & (gradient < 0)))
        direction = np.zeros_like(x)
        indices = np.flatnonzero(free)
        block = hessian[np.ix_(indices, indices)] + damping * np.diag(diagonal[indices])
        direction[indices] = np.linalg.solve(block, -gradient[indices])
        alternatives = (
            ("gauss_newton", direction),
            ("projected_gradient", -gradient / diagonal),
        )
        accepted = False
        for step_kind, direction in alternatives:
            for backtrack in range(18):
                candidate = np.clip(x + 0.5**backtrack * direction, lower, upper)
                slope = gradient @ (candidate - x)
                if slope >= 0:
                    continue
                if evaluations >= budget:
                    break
                trial = objective.evaluate(candidate, curvature=True)
                evaluations += 1
                hessian_evaluations += 1
                if trial[0] < value and trial[0] <= value + 1e-4 * slope:
                    x = candidate
                    value, gradient, hessian = trial
                    if step_kind == "gauss_newton" and backtrack == 0:
                        damping = max(1e-8, damping * 0.4)
                    else:
                        damping = min(1e8, damping * 4)
                    accepted = True
                    break
            if accepted or evaluations >= budget:
                break
        if not accepted:
            if evaluations < budget:
                status = "line search could not find sufficient decrease"
            break
    return OptimizeResult(
        x=x, fun=value, jac=gradient, nfev=evaluations, njev=evaluations,
        nhev=hessian_evaluations, nit=len(history) - 1, history=history,
        success=norm <= tolerance, message=status,
    )
