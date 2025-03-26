from typing import Optional
import tyro
from JaxSeq.bucket_manager import open_with_bucket as open
from JaxSeq.utils import convert_path, load_mesh, setup_experiment_save
import jax
import jax.numpy as jnp
from JaxSeq.utils import BlockingStrategy, Padding, Truncation, get_weight_decay_mask
import os
import optax
from JaxSeq.models.gpt2.load import load_train_state, ModelLoadMode
import pickle as pkl
from transformers.generation import GenerationConfig
from jaxtyping import PyTree
import re
from LLM_RL.environment import Text, TextTrajectory, TextTrajectoryChain, TokenTrajectoryChain
from LLM_RL.algorithms.mc_returns.data import MCData, MCDataset, MCIterableDataset
from LLM_RL.algorithms.value_rl_base.gpt2.interface import GPT2ValuePolicy
from LLM_RL.heads.mlp_head import load_train_state_from_config as load_head_train_state_from_config
from LLM_RL.heads.mlp_head import MLPHeadConfig
from LLM_RL.algorithms.mc_returns.gpt2.interface import GPT2MCTrain, GPT2MCInference
from functools import partial
from JaxSeq.logs import pull_logs
import json
from transformers import GPT2TokenizerFast
from llm_rl_scripts.chess.env.env import text_env_eval_chess_positions
from JaxSeq.shard_model import copy_sharded_pytree
import random 
from LLM_RL.algorithms.mc_returns.base_interface import mc_loss
from LLM_RL.algorithms.mc_returns.train import eval_loss, train_loop
from LLM_RL.algorithms.mc_returns.data import MCData, MCDataset

def main(
    model_load_mode: ModelLoadMode, 
    model_load_path: str, 
    train_data_path: str, 
    eval_data_path: str,
    test_positions_path: str,

    /,  # Mark the end of positional arguments.

    exp_name: Optional[str]=None, 
    outputs_path: Optional[str]=None, 

    data_mesh_shape: int=1, 
    fsdp_mesh_shape: int=1, 
    model_mesh_shape: int=-1, 

    use_wandb: bool=True, 
    wandb_project: Optional[str]="llm_rl_repo_give_position_ilql", 

    n_rounds: int=1, 
    epochs: int=1, 
    max_steps: Optional[int]=None, 
    
    lr: float=1e-4, 
    weight_decay: float=0.0, 
    tau: float=0.95,
    cql_weight: float=0.0,
    gamma: float=0.99,

    train_bsize: int=32, 
    grad_accum_steps: int=1, 

    gradient_checkpointing: bool=False, 
    gradient_checkpointing_policy: str='nothing_saveable', 

    max_length: int=80, 

    log_every: int=256, 
    eval_every_steps: Optional[int]=10000, 
    eval_every_epochs: Optional[int]=None, 
    eval_at_beginning: bool=False, 
    eval_at_end: bool=True, 

    save_every_steps: Optional[int]=100000, 
    save_every_epochs: Optional[int]=None, 
    save_at_beginning: bool=True, 
    save_at_end: bool=True, 
    save_best: bool=False, 
    max_checkpoints: Optional[int]=None, 
    save_train_state: bool=True, 
    save_bf16: bool=True, 

    policy_max_input_length: int=256, 
    policy_max_output_length: int=256, 
    policy_do_sample: bool=True, 
    policy_num_beams: int=1, 
    policy_temperature: Optional[float]=None, 
    policy_top_p: Optional[float]=None, 
    policy_top_k: Optional[int]=None, 

    force_pad_embeddings: bool=False, 

    should_restore_loop_state: bool=False, 
    reranker: bool=False,
):
    input_args = locals()
    print(input_args)

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    mesh = load_mesh((data_mesh_shape, fsdp_mesh_shape, model_mesh_shape), ('dp', 'fsdp', 'mp'))
    is_main_process = jax.process_index() == 0
    print(f"Mesh: {mesh}")
    print(f"Is main process: {is_main_process}")
    
    def mc_data_generator(data_name):
        with open(data_name, "r") as f:
            for item in f:
                obj = json.loads(item)
                # starting with the last element
                last_trajectory = TextTrajectory([Text(obj[-1]["state"], False), Text(obj[-1]["action"], True)], 
                                                 [0, obj[-1]["reward"]], True)
                curr_chain = TextTrajectoryChain(text_trajectory=last_trajectory, next=None)
                # curr_chain.next = curr_chain
                for traj in reversed(obj): # iterate through move history backwards except for last transition
                    # embed()
                    prev_trajectory = TextTrajectory([Text(traj["state"], False), Text(traj["action"], True)], 
                                                     [0, traj["reward"]], traj["done"])
                    curr_chain = TextTrajectoryChain(text_trajectory=prev_trajectory, next=curr_chain)
                token_trajectory_chain = TokenTrajectoryChain.from_text_trajectory_chain(curr_chain, tokenizer)
                while token_trajectory_chain.next is not None:
                    yield MCData.from_token_trajectory_chain(token_trajectory_chain, gamma=gamma)
                    token_trajectory_chain = token_trajectory_chain.next

    mc_data_lst = list(mc_data_generator(train_data_path))
    random.shuffle(mc_data_lst)
    
    dataset = MCIterableDataset.from_mc_data_iterable(mc_data_generator(train_data_path), tokenizer,
                                          BlockingStrategy(
                                                padding=Padding.RIGHT,
                                                truncation=Truncation.RIGHT,
                                                max_length=max_length,
                                            ))
    
    eval_dataset = MCIterableDataset.from_mc_data_iterable(mc_data_generator(eval_data_path), tokenizer,
                                            BlockingStrategy(
                                                padding=Padding.RIGHT,
                                                truncation=Truncation.RIGHT,
                                                max_length=max_length,
                                            ))
    
    def policy_optim_getter(params: PyTree):
        mask = get_weight_decay_mask((
            "".join([r"\['ln_[0-9]+'\]", re.escape("['bias']")]), 
            "".join([r"\['ln_[0-9]+'\]", re.escape("['scale']")]), 
            re.escape("['ln_f']['bias']"), 
            re.escape("['ln_f']['scale']"), 
            "bias", 
        ))(params)
        return optax.MultiSteps(
            optax.adamw(
                learning_rate=lr, 
                b1=0.9, 
                b2=0.95, 
                eps=1e-8, 
                weight_decay=weight_decay, 
                mask=mask, 
            ), 
            every_k_schedule=grad_accum_steps, 
        )
    
    def value_head_optim_getter(params: PyTree):
        mask = get_weight_decay_mask(("bias",))(params)
        return optax.MultiSteps(
            optax.adamw(
                learning_rate=lr, 
                b1=0.9, 
                b2=0.95, 
                eps=1e-8, 
                weight_decay=weight_decay, 
                mask=mask, 
            ), 
            every_k_schedule=grad_accum_steps, 
        )

    model_prng_key = jax.random.PRNGKey(3)
    base_train_state, base_model = load_train_state(
        model_load_mode=model_load_mode, 
        model_load_path=convert_path(model_load_path) if model_load_mode != ModelLoadMode.HF else model_load_path, 
        model_dtype=jnp.float32, 
        optim_getter=policy_optim_getter, 
        tokenizer=tokenizer, 
        mesh=mesh, 
        prng_key=model_prng_key, 
        force_pad_embeddings=force_pad_embeddings, 
        params_dtype=jnp.float32, 
    )
    base_model.config.gradient_checkpointing = gradient_checkpointing
    base_model.config.gradient_checkpointing_policy = gradient_checkpointing_policy
    pi_beta_params = copy_sharded_pytree(
        model=base_model, 
        pytree=base_train_state.params, 
    )

    q_prng_key = jax.random.PRNGKey(4)
    # embed()
    q_head_train_state, q_head = load_head_train_state_from_config(
        model_config=MLPHeadConfig(
            input_dim=base_model.config.n_embd, 
            hidden_dim=base_model.config.n_embd, 
            output_dim=base_model.config.vocab_size, 
            use_bias=True, 
            layer2_initializer_range=0.0, 
            layer2_bias_init=0.0, 
        ), 
        model_dtype=jnp.float32, 
        optim_getter=value_head_optim_getter, 
        mesh=mesh, 
        prng_key=q_prng_key, 
        pad_to_output_dim=None, 
        params_dtype=jnp.float32, 
    )

    loop_state = dict()
    if should_restore_loop_state and (model_load_mode in {ModelLoadMode.TRAIN_STATE, 
                                                          ModelLoadMode.TRAIN_STATE_PARAMS, 
                                                          ModelLoadMode.PARAMS}):
        with open(os.path.join(convert_path(model_load_path), 'loop_state.pkl'), 'rb') as f:
            loop_state = pkl.load(f)
    
    loss_fn = partial(mc_loss, cql_weight=cql_weight)

    train = GPT2MCTrain.load_train(
        base_train_state=base_train_state, 
        q_head_train_state=q_head_train_state, 
        base_model=base_model, 
        q_head_model=q_head, 
        tokenizer=tokenizer, 
        loss_fn=loss_fn, 
        detach_q=False, 
    )

    inference = GPT2MCInference.load_inference(
        pi_beta_params=pi_beta_params, 
        base_params=base_train_state.params, 
        q_head_params=q_head_train_state.params, 
        pi_beta_model=base_model, 
        base_model=base_model, 
        q_head_model=q_head, 
        tokenizer=tokenizer, 
        loss_fn=loss_fn, 
        beta=8.0, 
        dp_shard_logits=True, 
    )

    save_dir, exp_name = setup_experiment_save(
        exp_name=exp_name, 
        outputs_path=convert_path(outputs_path), 
        input_args=input_args, 
        script__file__=__file__, 
        is_main_process=is_main_process, 
    )

    with open(test_positions_path, "r") as f:
        positions = list(f)
        positions = [position.replace("\n", "").replace("\"", "") for position in positions if position != ""]
    
    positions = positions[:100]
        
    policy_prng = jax.random.PRNGKey(0)
    def evaluate(inference: GPT2MCInference):
        nonlocal policy_prng
        policy_prng, new_key = jax.random.split(policy_prng)
        policy = GPT2ValuePolicy(
                inference=inference.value_inference, 
                prng_key=new_key, 
                generation_config=GenerationConfig(
                    do_sample=True, 
                    num_beams=policy_num_beams, 
                    temperature=policy_temperature, 
                    top_p=policy_top_p, 
                    top_k=policy_top_k, 
                    eos_token_id=tokenizer.encode('\n')[0], 
                    pad_token_id=tokenizer.pad_token_id, 
                    max_new_tokens=policy_max_output_length, 
                ), 
                blocking_strategy=BlockingStrategy(
                    padding=Padding.LEFT, 
                    truncation=Truncation.LEFT, 
                    max_length=policy_max_input_length, 
                ), 
                out_str_process=lambda x: x.removesuffix('\n')+'\n', 
            )
        
        raw_results, summary_results = text_env_eval_chess_positions(
            positions=positions,
            policy=policy,
            n_rollouts=1,
            bsize=1,
        )
        
        data_results = eval_loss(
            inference=inference, 
            dataset=dataset, 
            prng_key=jax.random.PRNGKey(1), 
            bsize=4, 
            eval_batches=64, 
        )
        summary_results = pull_logs(summary_results)
        
        return data_results['losses']["total_loss"], {"interaction_env": summary_results}
                
            
        
    
    train_prng = jax.random.PRNGKey(1)
    save_dtype = jnp.bfloat16 if save_bf16 else jnp.float32
    trainer, inference = train_loop(
        trainer=train, 
        inference=inference, 
        evaluator=evaluate, 
        dataset=dataset, 
        prng_key=train_prng, 
        save_dir=save_dir, 
        n_rounds=n_rounds, 
        epochs=epochs, 
        max_steps=max_steps, 
        bsize=train_bsize, 
        log_every=log_every, 
        eval_every_steps=eval_every_steps, 
        eval_every_epochs=eval_every_epochs, 
        eval_at_beginning=eval_at_beginning, 
        eval_at_end=eval_at_end, 
        save_every_steps=save_every_steps, 
        save_every_epochs=save_every_epochs, 
        save_at_beginning=save_at_beginning, 
        save_at_end=save_at_end, 
        save_best=save_best, 
        max_checkpoints=max_checkpoints, 
        save_train_state=save_train_state, 
        save_dtype=save_dtype, 
        use_wandb=use_wandb, 
        wandb_project=wandb_project, 
        wandb_run_name=exp_name, 
        wandb_config=None, 
        is_main_process=is_main_process, 
        **loop_state, 
    )

if __name__ == "__main__":
    tyro.cli(main)
