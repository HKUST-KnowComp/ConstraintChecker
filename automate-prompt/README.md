# Automatic Prompt Augmentation and Selection with Chain-of-Thought from Labeled Data

[Original repo](https://github.com/SHUMKASHUN/Automate-CoT)

## 代码结构
- datasets: 数据集。数据集格式如下。其中prompt是一个要被填空的句子，每个空（大括号）按照targets被填。
    ```
    [
    {
        "prompt": "Q: Dragon is from the Greek word \"Lawrence.\"\nA: {}",
        "targets": [
        "NOT ENOUGH"
        ]
    },
    ...
    ]
    ```

- scripts: 执行模型的脚本。
- src: 模型。模型主要实现在roberta_eselect_openai.py。data里是数据集的实现，可以自己改。

## 执行
1. 执行入口是 `train_roberta_eselect.py`。
2. 由于一些奇怪的原因，在你的电脑上执行脚本前，请先修改train_roberta_eselect.py的12行：
    ```
    sys.path.append('./example-prompt')
    ```
   改成你的机器上项目文件夹的位置。

3. 执行时OpenAI API key通过环境变量 `OPENAI_API_KEY` 传入。

4. 一些应该有用的参数：
   - `entity`, `project_name`, `run_name`：传入 `enable_wandb` 时生效，是你的wandb的相应名称。
   - `knowledge_data_path`, `valid_data_path`：训练集和valid集位置。
   - `batch_size`, `lr`, `max_epochs`：你知道的
   - `model_name`：调用的OpenAI模型名称，默认最便宜的text-ada-001
   - `sample_size`：加几个example作为prompt呢？默认6
   - `pge_avg_samples`：按照PGE，我们要算好几次取平均。这个参数就是算几次

5. PyTorch Lightning的Trainer自带别的参数，来[查](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-class-api)。要使用GPU加速有些必须要写。举个例子：
   ```
   OPENAI_API_KEY=sk-不给你看 python3 scripts/train_roberta_eselect.py --accelerator gpu --strategy ddp --devices 2 --sample_size=6 --pge_avg_samples=3 --max_epochs=5
   ```

   不过这个代码可能用DDP没什么意义，因为你迟早要碰到rate limit
