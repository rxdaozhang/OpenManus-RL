import os
import json
import time
import jax
import pickle as pkl
from llm_rl_scripts.twenty_questions.env.policies import twenty_questions_env, UserPolicy
from LLM_RL.environment import text_env_eval

from JaxSeq.bucket_manager import open_with_bucket as open
from JaxSeq.utils import create_path
from LLM_RL.utils import convert_path

INTRO_TEXT = """\
Welcome to the game of Twenty Questions!
Your objective is to guess what the object is within twenty questions.
At every turn, you will have the oppurnity to ask a yes/no question, and receive an answer from the oracle.
You can ask tweny questions but must ask as few questions as possible. 

Now that you know the rules, let's get started. 
""".strip()

if __name__ == "__main__":
    YOUR_NAME = input("Enter your name: ").strip()
    N_INTERACTIONS = int(input("Enter number of trials: "))
    OUTPUTS_PATH = f"data/outputs/twenty_questions/human_eval/user_interactions_test1_{YOUR_NAME}/"
    env_deterministic = False
    print("Loading Environment")
    env = twenty_questions_env()
    print("Loaded Environment")
    policy = UserPolicy()

    def print_interaction(interaction):
        if interaction[-1].reward == 0:
            print('YOU WON!')
            print('='*25)
        else:
            print('YOU LOST :(')
            print('='*25)

    print()
    print('='*25)
    print(INTRO_TEXT)
    print("Here are the list of words you can select from:")
    words = [(word[0], word[1]) if len(word) > 1 else (word[0]) for word in env.word_list if len(word) >= 1]
    print()
    print(words)
    print('='*25)
    
    interation_raw_results, interaction_summary_results = text_env_eval(
        env=env,
        policy=policy,
        n_rollouts=N_INTERACTIONS,
        interaction_callback=print_interaction,
        env_options={"deterministic": env_deterministic}
    )

    print(interaction_summary_results)
    print('='*25)

    create_path(OUTPUTS_PATH)
    with open(os.path.join(OUTPUTS_PATH, 'interactions.pkl'), 'wb') as f:
        pkl.dump(interation_raw_results, f)
    with open(os.path.join(OUTPUTS_PATH, 'interactions_summary.json'), 'w') as f:
        json.dump(jax.tree_util.tree_map(lambda x: float(x), interaction_summary_results), f)


