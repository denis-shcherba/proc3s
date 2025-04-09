from __future__ import annotations

import os
import cma
import time
import pathlib
import logging
import importlib
import traceback
from typing import Callable, List, Union
import matplotlib.pyplot as plt
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

import numpy as np

from vtamp.environments.utils import Action, Environment, State
from vtamp.policies.utils import (
    Policy,
    Sampler,
    parse_code,
    query_llm,
)
from vtamp.utils import parse_text_prompt, save_log, write_prompt

_, _ = Action(), State()
log = logging.getLogger(__name__)


FUNC_NAME = "gen_plan"
FUNC_DOMAIN = "gen_domain"


def bbo_on_motion_plan(
    env: Environment,
    initial_state: State,
    plan_gen: Callable[[List[Union[int, float]]], List[Action]],
    domains_gen: List[Sampler],
    max_evals: int = 100,
) -> Union[List[Action], str]:
    
    failure_message = ""
    bbo_evals = 0

    domains = domains_gen(initial_state)

    #################################
    evals_infeasible = []
    evals_feasible = []
    #################################
    
    def compute_cost(input_vec: np.ndarray) -> float:
        env.reset()
        # env.C.view(True, "Before")
        ground_plan = plan_gen(initial_state, *input_vec)
        for action in ground_plan:
            env.step(action, vis=False)
        cost = env.compute_cost()
        #################################
        if cost == 1.11:
            evals_infeasible.append(input_vec)
        else:
            evals_feasible.append(input_vec)
        # env.C.view(True, f"After (cost: {cost}, guess: {input_vec}, eval_num: {len(evals_feasible)+len(evals_infeasible)})")
        print(f"Current Cost: {cost}")
        #################################
        return cost
    
    cma_initial_state = []
    for k, v in domains.items():
        cma_initial_state.append(v)

    bbo_options = {
        'popsize': 20,        # Number of candidate solutions per generation
        'maxiter': 500,       # Maximum number of generations/iterations
        'maxfevals': 100,   # Maximum number of function evaluations
        'tolfun': 1e-4,       # Stop if the change in function value is small
        'tolx': 1e-5,         # Stop if the step size (sigma) is very small
        'CMA_elitist': True,      # Keep the best from last generation
    }
    # sigma0: initial standard deviation
    result = cma.fmin(compute_cost, cma_initial_state, sigma0=.4, options=bbo_options)
    ground_plan = plan_gen(initial_state, *result[0])
    
    #################################
    print(f"Total feasible evals: {len(evals_feasible)}")
    print(f"Total infeasible evals: {len(evals_infeasible)}")
    print(f"Best solution: {result[0]}")
    print(f"Best cost: {result[1]}")
    print(f"Final Cost (Should be the same as the best cost): {compute_cost(result[0])}")

    eif = np.array(evals_infeasible).T
    ef = np.array(evals_feasible).T
    if len(eif):
        plt.scatter(eif[0], eif[1], label="infeasible")
    if len(ef):
        plt.scatter(ef[0], ef[1], label="feasible")
    plt.scatter([.3], [.3], label="target")
    plt.legend()
    plt.savefig("plot.png")
    #################################
    
    return ground_plan, failure_message, bbo_evals


def import_constants_from_class(cls):
    # Get the module name from the class
    module_name = cls.__module__

    # Dynamically import the module
    module = importlib.import_module(module_name)

    # Import all uppercase attributes (assuming these are constants)
    for attribute_name in module.__all__:
        # Importing the attribute into the global namespace
        globals()[attribute_name] = getattr(module, attribute_name)
        print(f"Imported {attribute_name}: {globals()[attribute_name]}")


class DENECK(Policy):
    def __init__(
        self,
        twin=None,
        max_feedbacks=0,
        seed=0,
        max_evals=0,
        use_cache=True,
        **kwargs,
    ):
        self.twin = twin
        self.seed = seed
        self.max_feedbacks = max_feedbacks
        self.max_evals = max_evals

        self.use_cache = use_cache

        import_constants_from_class(twin.__class__)

        # Get environment specific prompt
        prompt_fn = "prompt_{}".format(twin.__class__.__name__)
        prompt_path = os.path.join(
            pathlib.Path(__file__).parent, "{}.txt".format(prompt_fn)
        )

        self.prompt = parse_text_prompt(prompt_path)

        self.plan = None

    def get_action(self, belief, goal: str):
        statistics = {}
        if self.plan is None:
            # No plan yet, we need to come up with one
            ground_plan, statistics = self.full_query_bbo(belief, goal)

            if ground_plan is None:
                return None, statistics
            else:
                log.info("Found plan: {}".format(ground_plan))
                self.plan = ground_plan[1:]
                return ground_plan[0], statistics

        elif len(self.plan) > 0:
            next_action = self.plan[0]
            self.plan = self.plan[1:]
            return next_action, statistics

        return None, statistics
    
    def full_query_bbo(self, belief, task):
        self.twin.reset()
        content = "Goal: {}".format(task)
        content = "initial={}\n".format(str(belief)) + content
        chat_history = self.prompt + [{"role": "user", "content": content}]
        statistics = {}
        statistics["bbo_evals"] = 0
        statistics["bbo_solve_time"] = 0
        statistics["llm_query_time"] = 0
        statistics["num_bbo_evals"] = 0
        
        input_fn = "llm_input.txt"
        output_fn = "llm_output.txt"
        write_prompt(input_fn, chat_history)
        
        # TODO: use_cache
        # llm_response, statistics["llm_query_time"] = query_llm(chat_history, seed=0)
        #####################################################
        llm_response = open("./romanI_hlvlsr_bbo.txt", 'r').read()
        llm_query_time = 0
        #####################################################
        
        statistics["llm_query_time"] += llm_query_time
        
        chat_history.append({"role": "assistant", "content": llm_response})
        save_log(output_fn, llm_response)

        error_message = None
        ground_plan = None

        try:
            llm_code = parse_code(llm_response)
            exec(llm_code, globals())
            func = globals()[FUNC_NAME]
            if FUNC_DOMAIN in globals():
                domain = globals()[FUNC_DOMAIN]
                ground_plan, failure_message, bbo_evals = bbo_on_motion_plan(
                    self.twin,
                    belief,
                    func,
                    domain,
                    max_evals=self.max_evals,
                )
            else:
                log.info("No variables given to optimize. Continuing without BBO.")
                ground_plan = func(belief)

        except Exception as e:
            # Get the traceback as a string
            error_message = traceback.format_exc()
            log.info("Code error: " + str(error_message))

        # TODO: Add feedback response if there is an error
        
        return ground_plan, statistics
