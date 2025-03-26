
# OpenManus-RL 项目开发文档

## 项目概述

OpenManus-RL 是一个用于训练大型语言模型（LLM）智能体的强化学习框架，基于 RAGEN 和 Search-R1 项目的架构设计。该框架支持多种智能体框架、多个数据集和新的强化学习算法。

## 系统架构

### 核心组件

```
openmanus_rl/
├── llm_agent/                  # 智能体框架实现
├── algos/                   # 强化学习算法实现 
└── envs/                # 环境实现
```

## 详细设计

### 1. 环境抽象层（可直接复用 RAGEN 的设计）

```python
# openmanus_rl/agentgym/base_env.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

class BaseEnv(ABC):
    """所有环境的抽象基类"""
    def __init__(self):
        self.reward = 0
        self._actions = []        # 所有动作列表
        self._actions_valid = []  # 格式正确的动作列表
        self._actions_effective = [] # 有效动作列表
        
    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Any:
        """重置环境"""
        pass

    @abstractmethod
    def step(self, action) -> Tuple[Any, float, bool, Dict]:
        """执行动作并返回观察、奖励、终止标志和额外信息"""
        pass
    
    def get_tracking_variables(self) -> Dict:
        """获取环境跟踪变量"""
        return {
            "actions": self._actions,
            "actions_valid": self._actions_valid,
            "actions_effective": self._actions_effective,
            "reward": self.reward
        }
```

### 2. 环境实现（需要为每个支持的数据集开发）

```python
# openmanus_rl/agentgym/gaia_env.py
from openmanus_rl.agentgym.base_env import BaseEnv

class GAIAEnv(BaseEnv):
    """GAIA环境实现"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 初始化GAIA环境特定属性
        
    def reset(self, seed=None):
        # 重置环境
        pass
        
    def step(self, action):
        # 执行动作并返回反馈
        pass
```

同样需要为其他数据集实现环境类：
- `openmanus_rl/agentgym/webshop_env.py`
- `openmanus_rl/agentgym/agenttuning_env.py`
- `openmanus_rl/agentgym/agentcompany_env.py`

### 3. 环境工厂（新组件，用于统一管理环境）

```python
# openmanus_rl/agentgym/env_factory.py
from openmanus_rl.agentgym.gaia_env import GAIAEnv
from openmanus_rl.agentgym.webshop_env import WebShopEnv
from openmanus_rl.agentgym.agenttuning_env import AgentTuningEnv
from openmanus_rl.agentgym.agentcompany_env import AgentCompanyEnv

class EnvFactory:
    """环境工厂类，用于创建不同类型的环境"""
    @staticmethod
    def create_env(env_type, config):
        if env_type == "gaia":
            return GAIAEnv(config)
        elif env_type == "webshop":
            return WebShopEnv(config)
        elif env_type == "agenttuning":
            return AgentTuningEnv(config)
        elif env_type == "agentcompany":
            return AgentCompanyEnv(config)
        else:
            raise ValueError(f"不支持的环境类型: {env_type}")
```

### 4. 生成管理器（可复用 RAGEN 的设计并扩展）

```python
# openmanus_rl/training/llm_generation_manager.py
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
import torch

from openmanus_rl.models.data_proto import DataProto

@dataclass
class GenerationConfig:
    max_turns: int             # 最大交互轮次
    max_start_length: int      # 初始输入最大长度
    max_prompt_length: int     # 提示词最大长度
    max_response_length: int   # 响应最大长度
    max_obs_length: int        # 观察值最大长度
    no_think_rl: bool=False    # 是否不进行思考
    agent_framework: str=None  # 智能体框架类型

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        env_class,
        config: GenerationConfig,
        logger=None,
        is_validation: bool = False,
    ):
        # 初始化组件
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.env_class = env_class
        self.config = config
        self.logger = logger
        self.is_validation = is_validation
        self.tensor_fn = TensorFn(tokenizer.pad_token_id)
    
    def run_llm_loop(self, envs, initial_input_ids):
        """运行主要的LLM与环境交互循环"""
        # 初始化状态
        # 循环交互直到终止条件
        # 收集轨迹数据
        # 几乎可以直接复用RAGEN的实现
        pass
```

### 5. 强化学习算法实现（扩展 RAGEN 的实现）

```python
# openmanus_rl/algos/advantage_estimators.py
import torch
from typing import Tuple

def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    eos_mask: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算广义优势估计（GAE）
    """
    # 从RAGEN或SearchR1复用实现
    pass

def compute_grpo_advantage(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算GRPO优势估计
    """
    # 从RAGEN复用实现
    pass

def compute_gapo_advantage(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    eos_mask: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
    group_size: int = 16
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算GAPO（Group-Aware Policy Optimization）优势估计
    """
    # 实现新的GAPO算法
    pass
```

### 6. 智能体框架实现（新组件）

创建专门处理不同智能体框架的模块：

```python
# openmanus_rl/agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any

class BaseAgent(ABC):
    """智能体抽象基类"""
    
    @abstractmethod
    def process_input(self, observation, history=None):
        """处理输入观察"""
        pass
    
    @abstractmethod
    def process_output(self, model_output):
        """处理模型输出转换为环境动作"""
        pass
    
    @abstractmethod
    def get_format_template(self):
        """获取该智能体框架的格式模板"""
        pass
```

分别实现不同的智能体框架：

```python
# openmanus_rl/agents/react_agent.py
from openmanus_rl.agents.base_agent import BaseAgent

class ReActAgent(BaseAgent):
    """实现ReAct框架的智能体"""
    
    def process_input(self, observation, history=None):
        # 处理输入
        pass
    
    def process_output(self, model_output):
        # 处理输出，提取思考过程和动作
        pass
    
    def get_format_template(self):
        return """
        以下是ReAct格式的示例:
        <think>
        我需要思考如何解决这个问题...
        </think>
        <action>执行的动作</action>
        """
```

同样需要实现其他智能体框架：
- `openmanus_rl/agents/reflexion_agent.py`
- `openmanus_rl/agents/cot_agent.py`
- `openmanus_rl/agents/tool_agent.py`

### 7. 智能体工厂（新组件）

```python
# openmanus_rl/agents/agent_factory.py
from openmanus_rl.agents.react_agent import ReActAgent
from openmanus_rl.agents.reflexion_agent import ReflexionAgent
from openmanus_rl.agents.cot_agent import CoTAgent
from openmanus_rl.agents.tool_agent import ToolAgent

class AgentFactory:
    """智能体工厂类，用于创建不同类型的智能体框架"""
    @staticmethod
    def create_agent(agent_type, config=None):
        if agent_type == "react":
            return ReActAgent(config)
        elif agent_type == "reflexion":
            return ReflexionAgent(config)
        elif agent_type == "cot":
            return CoTAgent(config)
        elif agent_type == "tool":
            return ToolAgent(config)
        else:
            raise ValueError(f"不支持的智能体类型: {agent_type}")
```

### 8. 数据集处理模块

```python
# openmanus_rl/data/dataset.py
import pandas as pd
from torch.utils.data import Dataset

class RLHFDataset(Dataset):
    """强化学习数据集"""
    def __init__(
        self,
        parquet_files,
        tokenizer,
        prompt_key="prompt",
        max_prompt_length=2048,
        filter_prompts=True,
        return_raw_chat=False,
        truncation="error"
    ):
        # 初始化数据集
        # 可以复用Search-R1的实现
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.prompt_key = prompt_key
        self.filter_prompts = filter_prompts
        self.return_raw_chat = return_raw_chat
        self.truncation = truncation
        
        # 加载数据
        dataframes = []
        for parquet_file in parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # 获取数据项
        # 处理提示和奖励信息
        # 可以复用Search-R1的实现
        pass
```

### 9. 训练流程实现

```python
# openmanus_rl/training/trainer.py
import os
import torch
from torch.utils.data import DataLoader

from openmanus_rl.models.data_proto import DataProto
from openmanus_rl.agentgym.env_factory import EnvFactory
from openmanus_rl.agents.agent_factory import AgentFactory
from openmanus_rl.training.llm_generation_manager import LLMGenerationManager, GenerationConfig

class RLTrainer:
    """强化学习训练器"""
    def __init__(self, config, tokenizer, model, reward_fn=None):
        self.config = config
        self.tokenizer = tokenizer
        self.model = model
        self.reward_fn = reward_fn
        
        # 创建环境
        self.env_class = EnvFactory.create_env(config.env.type, config.env)
        
        # 创建智能体
        self.agent = AgentFactory.create_agent(config.agent.type, config.agent)
        
        # 创建生成管理器
        self.gen_manager = LLMGenerationManager(
            tokenizer=tokenizer,
            actor_rollout_wg=model,
            env_class=self.env_class,
            config=GenerationConfig(
                max_turns=config.training.max_turns,
                max_start_length=config.data.max_start_length,
                max_prompt_length=config.data.max_prompt_length,
                max_response_length=config.data.max_response_length,
                max_obs_length=config.data.max_obs_length,
                no_think_rl=config.algorithm.no_think_rl,
                agent_framework=config.agent.type
            ),
            logger=None,
            is_validation=False
        )
        
    def train(self):
        """执行训练流程"""
        # 加载数据集
        # 创建数据加载器
        # 执行PPO训练循环
        # 保存模型和结果
        pass
```

### 10. 配置文件结构

```yaml
# configs/base_config.yaml
data:
  train_files: ["data/gaia/train.parquet"]
  val_files: ["data/gaia/val.parquet"]
  prompt_key: "prompt"
  max_prompt_length: 2048
  max_response_length: 1024
  max_obs_length: 512
  max_start_length: 4096
  train_batch_size: 8
  shuffle_train_dataloader: true

env:
  type: "gaia"  # gaia, webshop, agenttuning, agentcompany
  max_steps: 20

agent:
  type: "react"  # react, reflexion, cot, tool

model:
  base_model: "Qwen/Qwen2.5-0.5B-Instruct"
  experiment_name: "openmanus_experiment"

training:
  micro_batch_size: 1
  use_kl_loss: true
  max_turns: 5
  n_rollout: 16
  train_batch_size: 8
  ppo_batch_size: 128
  no_think_rl: false

optimization:
  actor_lr: 1e-6
  critic_lr: 1e-5
  kl_coef: 0.001
  kl_loss_type: "low_var_kl"
  adv_estimator: "grpo"  # gae, grpo, gapo
```

## 开发路线图

### 阶段一：核心框架

1. 实现基础环境抽象类
2. 实现智能体框架抽象类
3. 实现数据集加载模块
4. 实现基本训练流程

### 阶段二：环境支持

1. 实现GAIA环境
2. 实现Webshop环境
3. 实现AgentTuning环境
4. 实现AgentCompany环境

### 阶段三：智能体框架

1. 实现ReAct框架
2. 实现Reflexion框架
3. 实现CoT框架
4. 实现Tool框架

### 阶段四：算法扩展

1. 实现GRPO算法
2. 实现GAPO算法
3. 优化奖励函数

## 代码复用策略

1. 从RAGEN复用：
   - 环境抽象类设计
   - LLM生成管理器
   - GRPO算法实现

2. 从Search-R1复用：
   - 数据集加载和处理
   - PPO训练循环
   - 奖励计算

3. 最小量开发：
   - 专注于实现环境适配器
   - 实现智能体框架工厂
   - 配置文件系统

## 结论

OpenManus-RL项目可以在很大程度上复用RAGEN和Search-R1的架构设计和代码实现，主要的开发工作集中在：

1. 环境适配：为GAIA、Webshop等数据集实现环境接口
2. 智能体框架：实现不同的智能体框架如ReAct、Reflexion等
3. 训练算法：扩展GRPO并实现GAPO等新算法

通过工厂模式和抽象基类设计，可以灵活支持多种环境和智能体框架，同时保持代码结构清晰和易于扩展。
