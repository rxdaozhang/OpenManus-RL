# Search-R1 项目训练流程文档

## 项目概述

Search-R1是一个基于大型语言模型（LLM）的搜索增强智能体训练框架，使用近端策略优化（PPO）方法来训练能够执行搜索和回答操作的语言模型。项目利用VERL（Versatile Entity Reinforcement Learning）框架作为底层训练基础设施。

## 训练流程概述

以`train_ppo.sh`脚本为例，整个训练过程大致可以分为以下几个步骤：

1. 环境配置与初始化
2. 创建训练代理和相关组件
3. 执行PPO训练循环
4. 验证和保存模型

## 核心组件与类

### 1. 环境定义

环境主要通过`LLMGenerationManager`类实现，该类定义了模型如何与搜索环境交互：

```python
class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    )
```

配置项通过`GenerationConfig`类定义：

```python
@dataclass
class GenerationConfig:
    max_turns: int             # 最大对话轮次
    max_start_length: int      # 初始输入最大长度
    max_prompt_length: int     # 提示词最大长度
    max_response_length: int   # 响应最大长度
    max_obs_length: int        # 观察值最大长度
    num_gpus: int              # GPU数量
    no_think_rl: bool=False    # 是否不进行思考
    search_url: str = None     # 搜索URL
    topk: int = 3              # 搜索结果Top-K
```

### 2. 模型与环境交互

环境交互主要通过`execute_predictions`方法实现：

```python
def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True) -> List[str]:
    """执行预测并返回环境响应"""
    # 处理预测结果
    cur_actions, contents = self.postprocess_predictions(predictions)
    next_obs, dones, valid_action, is_search = [], [], [], []
    
    # 处理搜索请求
    search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
    if do_search:
        search_results = self.batch_search(search_queries)
    else:
        search_results = [''] * sum([1 for action in cur_actions if action == 'search'])
    
    # 根据动作类型返回不同结果
    for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
        if not active:
            # 处理非活跃实例
            next_obs.append('')
            dones.append(1)
            valid_action.append(0)
            is_search.append(0)
        else:
            if action == 'answer':
                # 回答操作，结束交互
                next_obs.append('')
                dones.append(1)
                valid_action.append(1)
                is_search.append(0)
            elif action == 'search':
                # 搜索操作，继续交互
                next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                dones.append(0)
                valid_action.append(1)
                is_search.append(1)
            else:
                # 无效操作，给予反馈
                next_obs.append(f'\nMy previous action is invalid...')
                dones.append(0)
                valid_action.append(0)
                is_search.append(0)
                
    return next_obs, dones, valid_action, is_search
```

### 3. PPO训练器

训练过程通过`RayPPOTrainer`类控制：

```python
class RayPPOTrainer(object):
    def __init__(self,
             config,
             tokenizer,
             role_worker_mapping: dict[Role, WorkerType],
             resource_pool_manager: ResourcePoolManager,
             ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
             reward_fn=None,
             val_reward_fn=None)
```

主要训练循环在`fit`方法中实现，大致流程为：

1. 初始化代理环境
2. 循环处理训练批次数据
3. 执行LLM交互过程（搜索->获取信息->决策）
4. 计算奖励和优势估计
5. 更新演员和评论家网络
6. 记录指标并验证

### 4. 奖励获取

奖励计算主要通过`RewardManager`类实现：

```python
class RewardManager():
    def __init__(self, tokenizer, num_examine, format_score=0.) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_score = format_score
        
    def __call__(self, data: DataProto):
        # 计算奖励值
```

具体的奖励计算方法主要使用精确匹配（Exact Match）来衡量生成答案与参考答案的匹配度：

```python
def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """计算精确匹配奖励"""
    answer = extract_solution(solution_str=solution_str)
    
    if answer is None:
        return 0
    else:
        if em_check(answer, ground_truth['target']):
            return score  # 完全匹配返回满分
        else:
            return format_score  # 不匹配返回格式分
```

### 5. PPO算法核心

PPO算法的核心计算在`apply_kl_penalty`和`compute_advantage`函数中实现：

```python
def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    """应用KL散度惩罚项"""
    # 计算引用策略和当前策略之间的KL散度
    kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'], kl_penalty=kl_penalty)
    beta = kl_ctrl.value
    token_level_rewards = token_level_scores - beta * kld
    # 更新KL控制器
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    
def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    """计算优势函数"""
    if adv_estimator == 'gae':
        # 广义优势估计（GAE）
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=token_level_rewards,
            values=values,
            eos_mask=response_mask,
            gamma=gamma,
            lam=lam
        )
    elif adv_estimator == 'grpo':
        # GRPO优势估计
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            eos_mask=response_mask,
            index=index
        )
```

## 搜索环境交互流程

1. 模型生成响应（`generate_sequences`）
2. 从响应中提取动作（`postprocess_predictions`）
3. 执行动作并获取环境反馈（`execute_predictions`）
   - 若动作为"search"，调用搜索API获取信息（`batch_search`）
   - 若动作为"answer"，结束对话并评估回答
4. 将反馈作为下一轮输入

## 训练过程关键点

1. **动作空间**：模型可执行的动作包括"search"（搜索）和"answer"（回答）两种操作
2. **状态表示**：通过提示词（prompt）和历史交互构建状态表示
3. **奖励机制**：主要基于回答的准确性（EM评分）计算奖励
4. **分布式训练**：使用Ray框架实现分布式PPO训练
5. **模型结构**：采用Actor-Critic架构，Actor生成动作，Critic评估状态值

## 总结

Search-R1项目通过PPO算法训练大型语言模型，使其能够学习何时进行搜索以及如何利用搜索结果回答问题。整个框架基于VERL构建，使用Ray进行分布式训练，通过精确匹配评分机制为模型提供学习信号。这种框架使模型能够在外部知识检索方面显著提升能力，更好地处理开放域问题回答任务。

## 环境设置与实现

Search-R1项目采用了一种轻量级的环境设计方式，并没有使用OpenAI Gym等标准强化学习环境库，而是通过自定义的检索服务和简单的动作-观察机制构建了交互环境。

### 1. 检索服务环境

项目使用独立的检索服务器作为环境的一部分，通过HTTP API与模型进行交互。检索服务通过`retrieval_server.py`实现，主要包含以下组件：

```python
# 启动检索服务的脚本
# retrieval_launch.sh
file_path=/the/path/you/save/corpus
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever=intfloat/e5-base-v2

python search_r1/search/retrieval_server.py --index_path $index_file \
                                           --corpus_path $corpus_file \
                                           --topk 3 \
                                           --retriever_model $retriever
```

检索服务支持两种主要检索方式：

1. **BM25检索器**：基于传统的词频-逆文档频率算法
   ```python
   class BM25Retriever(BaseRetriever):
       def __init__(self, config):
           super().__init__(config)
           from pyserini.search.lucene import LuceneSearcher
           self.searcher = LuceneSearcher(self.index_path)
   ```

2. **密集向量检索器**：基于预训练语言模型的语义检索
   ```python
   class DenseRetriever(BaseRetriever):
       def __init__(self, config):
           super().__init__(config)
           self.encoder = Encoder(
               model_name=config.retrieval_model_path,
               model_path=config.retrieval_model_path,
               pooling_method=config.retrieval_pooling_method,
               max_length=config.retrieval_query_max_length,
               use_fp16=config.retrieval_use_fp16
           )
   ```

### 2. 环境接口设计

环境接口通过FastAPI实现，提供了简单的REST API：

```python
@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """检索端点，接收查询请求并返回检索结果"""
    queries = request.queries
    topk = request.topk
    return_scores = request.return_scores
    
    result = []
    for query in queries:
        docs, scores = retriever.search(query, num=topk, return_score=True)
        result.append({"document": docs, "scores": scores if return_scores else None})
    
    return {"result": result}
```

### 3. 环境参数配置

训练脚本中通过以下参数配置检索环境：

```bash
# 从train_ppo.sh中的环境配置部分
retriever.url="http://127.0.0.1:8000/retrieve" \
retriever.topk=3 \
```

这些参数在`GenerationConfig`中被传递：

```python
gen_config = GenerationConfig(
    max_turns=self.config.max_turns,
    max_start_length=self.config.data.max_start_length,
    max_prompt_length=self.config.data.max_prompt_length,
    max_response_length=self.config.data.max_response_length,
    max_obs_length=self.config.data.max_obs_length,
    num_gpus=self.config.trainer.n_gpus_per_node,
    no_think_rl=self.config.algorithm.no_think_rl,
    search_url=self.config.retriever.url,
    topk=self.config.retriever.topk,
)
```

### 4. 环境与模型交互

环境交互不是通过标准的Gym接口，而是通过`LLMGenerationManager`类中的方法：

1. **批量搜索**：通过HTTP请求调用检索服务
   ```python
   def _batch_search(self, queries):
       payload = {
           "queries": queries,
           "topk": self.config.topk,
           "return_scores": True
       }
       return requests.post(self.config.search_url, json=payload).json()
   ```

2. **处理搜索结果**：将检索结果格式化为可读文本
   ```python
   def _passages2string(self, retrieval_result):
       format_reference = ''
       for idx, doc_item in enumerate(retrieval_result):
           content = doc_item['document']['contents']
           title = content.split("\n")[0]
           text = "\n".join(content.split("\n")[1:])
           format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
       return format_reference
   ```

### 5. 环境状态表示

环境状态主要通过提示词（prompt）和历史交互记录来表示。在每一轮交互中，模型输出被处理为"search"或"answer"动作，然后环境返回相应的信息或结束标志：

```python
# 对于搜索动作，环境返回检索结果
next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
# 对于回答动作，环境返回空字符串并设置done标志
next_obs.append('')
dones.append(1)
```

### 6. 状态更新与训练

环境状态更新主要在`run_llm_loop`方法中，它实现了完整的交互循环：

```python
def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
    """运行主要的LLM生成循环"""
    
    # 初始化左侧和右侧状态
    original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
    original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
    
    # 循环进行直到最大轮次或所有实例完成
    for step in range(self.config.max_turns):
        # 生成响应
        gen_output = self._generate_with_gpu_padding(rollings_active)
        responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
        
        # 执行预测并获取环境反馈
        next_obs, dones, valid_action, is_search = self.execute_predictions(
            responses_str, self.tokenizer.pad_token, active_mask
        )
        
        # 更新状态
        next_obs_ids = self._process_next_obs(next_obs)
        rollings = self._update_rolling_state(rollings, responses_ids, next_obs_ids)
        original_right_side = self._update_right_side(original_right_side, responses_ids, next_obs_ids)
```

这种环境设计简单而灵活，专为语言模型与检索系统的交互优化，使模型能够学习何时进行搜索以及如何根据检索结果提供更准确的回答。

## 数据格式与处理

### 数据格式

训练数据存储在Parquet文件中（如`data/nq_search/train.parquet`和`test.parquet`），主要包含以下关键字段：

1. **data_source**: 数据来源标识，如"nq"（Natural Questions数据集）
2. **prompt**: 问题提示，以聊天格式存储
   ```json
   "prompt": [{
       "role": "user",
       "content": "问题内容..."
   }]
   ```
3. **ability**: 任务能力标识，如"fact-reasoning"
4. **reward_model**: 奖励模型相关信息
   ```json
   "reward_model": {
       "style": "rule",
       "ground_truth": {"target": ["正确答案1", "正确答案2", ...]}
   }
   ```
5. **extra_info**: 额外信息，如数据分割("train"/"test")和索引编号

### 数据处理流程

1. **原始数据加载**：
   项目从公开数据集加载原始数据，例如"RUC-NLPIR/FlashRAG_datasets"中的"nq"数据集。

2. **问题格式化**：
   对问题进行格式化处理，添加特定的指令前缀：
   ```python
   prefix = f"""Answer the given question. 
   You must conduct reasoning inside <think> and </think> first every time you get new information. 
   After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> 
   and it will return the top searched results between <information> and </information>. 
   You can search as many times as your want. 
   If you find no further external knowledge needed, you can directly provide the answer 
   inside <answer> and </answer>, without detailed illustrations. 
   For example, <answer> Beijing </answer>. Question: {question}\n"""
   ```

3. **数据转换**：
   将原始数据转换为训练所需的结构化格式：
   ```python
   data = {
       "data_source": data_source,
       "prompt": [{
           "role": "user",
           "content": question,
       }],
       "ability": "fact-reasoning",
       "reward_model": {
           "style": "rule",
           "ground_truth": solution
       },
       "extra_info": {
           'split': split,
           'index': idx,
       }
   }
   ```

4. **数据保存**：
   处理后的数据以Parquet格式保存为训练集（train.parquet）和测试集（test.parquet）文件。

### 数据加载与处理

训练过程中，数据通过`RLHFDataset`类加载和处理：

1. **数据读取**：
   ```python
   dataframes = []
   for parquet_file in self.parquet_files:
       dataframe = pd.read_parquet(parquet_file)
       dataframes.append(dataframe)
   self.dataframe = pd.concat(dataframes)
   ```

2. **数据处理**：
   数据项在通过`__getitem__`方法被获取时会进行处理：
   - 使用模型的`chat_template`将聊天格式转换为模型可接受的格式
   - 将文本转换为token ID
   - 创建注意力掩码和位置ID
   ```python
   input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
       prompt=prompt_with_chat_template,
       tokenizer=self.tokenizer,
       max_length=self.max_prompt_length,
       pad_token_id=self.tokenizer.pad_token_id,
       left_pad=True,
       truncation=self.truncation
   )
   position_ids = compute_position_id_with_mask(attention_mask)
   ```

3. **批处理**：
   数据通过`collate_fn`函数被合并成批次，同时处理张量和非张量数据。

### 训练数据流程

在PPO训练中，数据流程如下：

1. **数据加载器创建**：
   ```python
   self.train_dataset = RLHFDataset(
       parquet_files=self.config.data.train_files,
       tokenizer=self.tokenizer,
       prompt_key=self.config.data.prompt_key,
       max_prompt_length=self.config.data.max_prompt_length,
       filter_prompts=True,
       return_raw_chat=self.config.data.get('return_raw_chat', False),
       truncation='error'
   )
   
   self.train_dataloader = DataLoader(
       dataset=self.train_dataset,
       batch_size=self.config.data.train_batch_size,
       shuffle=self.config.data.shuffle_train_dataloader,
       drop_last=True,
       collate_fn=collate_fn
   )
   ```

2. **数据迭代**：
   训练循环中通过迭代数据加载器获取批次数据：
   ```python
   for batch_dict in self.train_dataloader:
       batch: DataProto = DataProto.from_single_dict(batch_dict)
       # 进行训练...
   ```

这种数据格式设计使模型能够学习如何与搜索环境进行交互，允许模型通过`<search>`标签发起搜索请求，并通过`<answer>`标签提供最终答案。训练过程中使用的标准化数据格式确保了模型可以一致地理解任务要求和奖励信号。
