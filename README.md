# 实验10：DDPM扩散模型图像生成

本项目包含MNIST基础DDPM、CIFAR-10选做任务，以及以下扩展：

- 线性与余弦噪声调度对比。
- 类别条件DDPM与Classifier-Free Guidance（CFG）。
- DDIM快速采样与DDPM采样对比。

训练过程会保存完整配置、运行环境、检查点、CSV指标、loss曲线和逐epoch生成预览，便于在远程服务器运行后拉取结果撰写报告。

## 1. 环境配置

建议使用Python 3.9或更高版本。服务器应先按照CUDA版本安装官方PyTorch，再安装其余依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

确认GPU可用：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

运行项目环境和张量自检：

```bash
python code/check_environment.py --device cuda
```

## 2. 先执行快速验证

单元测试不下载数据：

```bash
python -m pytest -q
```

使用64张MNIST图像完成一次端到端训练：

```bash
python code/train_mnist_ddpm.py --config configs/mnist_quick.json
```

确认生成了`outputs/smoke_test/latest.pt`和预览图片后，再运行正式实验。

## 3. MNIST基础任务

训练无条件DDPM：

```bash
python code/train_mnist_ddpm.py
```

断点续训：

```bash
python code/train_mnist_ddpm.py --resume auto
```

生成实验要求的结果：

```bash
python code/show_mnist_ddpm.py
```

主要输出：

```text
outputs/mnist_baseline/latest.pt
outputs/mnist_baseline/metrics.csv
outputs/mnist_baseline/loss_curve.png
outputs/mnist_results/mnist_generated_grid.png
outputs/mnist_results/mnist_denoising_process.gif
outputs/mnist_results/mnist_denoising_process.png
outputs/mnist_results/mnist_forward_noise_label_x.png
```

## 4. 条件DDPM与CFG

先训练独立MNIST分类器，用于定量评价生成结果：

```bash
python code/train_mnist_classifier.py
```

训练默认使用余弦调度的条件DDPM：

```bash
python code/train_conditional_ddpm.py
```

生成按0至9排列的图片，并比较CFG强度：

```bash
python code/show_conditional_ddpm.py \
  --guidance-scales 0 1 2 4 \
  --selected-scale 2
```

输出包含条件生成网格、CFG对比图、分类准确率曲线、混淆矩阵和CSV指标。

## 5. 线性与余弦调度对比

为保证控制变量，使用相同配置额外训练线性调度模型：

```bash
python code/train_conditional_ddpm.py \
  --beta-schedule linear \
  --output-dir outputs/mnist_conditional_linear
```

执行对比：

```bash
python code/compare_schedules.py \
  --linear-checkpoint outputs/mnist_conditional_linear/latest.pt \
  --cosine-checkpoint outputs/mnist_conditional/latest.pt
```

将输出噪声曲线、loss曲线、同初始噪声生成对比和定量指标。

## 6. DDPM与DDIM采样对比

```bash
python code/compare_samplers.py \
  --checkpoint outputs/mnist_conditional/latest.pt \
  --ddim-steps 100 50 25
```

DDPM与各组DDIM使用相同初始噪声。脚本在计时前执行GPU预热，记录总耗时、单图耗时、条件准确率、置信度和像素多样性。

## 7. 参数消融实验

默认比较`EPOCHS`、`TIMESTEPS`和`MAX_TRAIN_SAMPLES`：

```bash
python code/run_ablation.py
python code/plot_experiments.py
```

先检查将要执行的命令：

```bash
python code/run_ablation.py --dry-run
```

测试批量调度而不进行长训练：

```bash
python code/run_ablation.py --quick
```

可增加模型宽度与batch size实验：

```bash
python code/run_ablation.py \
  --dimensions epochs timesteps max_train_samples base_channels batch_size
```

## 8. CIFAR-10选做任务

默认配置会读取：

```text
DDPM参考资料/data/cifar-10-python.tar.gz
```

也可以显式指定服务器上的压缩包：

```bash
python code/train_cifar10_ddpm.py \
  --cifar-archive /path/to/cifar-10-python.tar.gz
python code/show_cifar10_ddpm.py \
  --cifar-archive /path/to/cifar-10-python.tar.gz
```

若基础配置生成效果较差，使用完整数据集和更大模型重新训练高质量版本：

```bash
python code/train_cifar10_ddpm.py \
  --config configs/cifar10_quality.json \
  --cifar-archive /path/to/cifar-10-python.tar.gz

python code/show_cifar10_ddpm.py \
  --checkpoint outputs/cifar10_quality/latest.pt \
  --output-dir outputs/cifar10_quality_results \
  --sampler ddim \
  --sampling-steps 250
```

该配置使用全部50,000张训练图、300轮、1,000个扩散步和64个基础通道。建议在24GB显存GPU上使用`batch_size=128`；显存不足时只减小batch size，不修改其他参数。

## 9. 一键运行

核心实验：

```bash
bash run_server_experiments.sh
```

同时执行参数实验和CIFAR-10：

```bash
RUN_ABLATION=1 RUN_CIFAR=1 bash run_server_experiments.sh
```

建议先分别运行各阶段，检查输出质量后再启动耗时较长的消融实验。

## 10. 常用参数覆盖

所有训练脚本都支持命令行覆盖JSON配置，例如：

```bash
python code/train_mnist_ddpm.py \
  --epochs 30 \
  --batch-size 256 \
  --timesteps 500 \
  --base-channels 48 \
  --max-train-samples 20000 \
  --device cuda
```

CUDA显存不足时，优先减小`--batch-size`；预览采样耗时过长时，可增大`--sample-every`或减小`--preview-sampling-steps`。

## 11. 拉取实验结果

需要完整保留`outputs/`目录，尤其是：

- `config.json`和`environment.json`
- `metrics.csv`和`training_summary.json`
- `*.png`与`*.gif`
- `latest.pt`或`best.pt`

服务器上可打包：

```bash
tar -czf ddpm_outputs.tar.gz outputs
```

下载并解压回本项目根目录后，即可基于真实训练数据生成实验报告。
