# Tianshou-MARL

Tianshou demos for multi-agent reinforcement learning.

Use the following command to clone the repository with submodules:

```bash
git clone --recurse-submodules --shallow-submodules --depth 1 https://github.com/liuzhaoze/Tianshou-MARL.git
```

## Python Environment Setup

Highly recommend to setup the environment on Ubuntu/macOS.

Install required build tools:

```bash
sudo apt install cmake swig zlib1g-dev
```

Create a new conda environment:

```bash
# Create a new conda environment
conda create -n tianshou python=3.11

# Activate the environment
conda activate tianshou
```

Install Tianshou:

```bash
# clone the Tianshou repository outside of this repository directory
git clone --branch v1.1.0 --depth 1 https://github.com/thu-ml/tianshou.git
cd tianshou
pip install poetry

# Change the source of poetry if necessary
poetry source add --priority=primary tsinghua https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

poetry lock --no-update
poetry install
```

Install `multi-agent-ale`:

```bash
# execute the following command after switching the working path to this repository.
cd multi-agent-ale
python setup.py install
```

Install PettingZoo environments:

```bash
# the file `constraints.txt` is located in the root directory of this repository
pip install 'pettingzoo[all]==1.24.2' --constraint constraints.txt
```

Install TensorBoard and [WandB](https://wandb.ai/home) for logging:

```bash
pip install tensorboard
pip install wandb
```

## Usage

### [Tic Tac Toe](https://pettingzoo.farama.org/environments/classic/tictactoe/)

```bash
python tianshou-tictactoe.py --watch --watch-episode-num 3
python tianshou-tictactoe.py --watch --watch-episode-num 3 --learned-go-first
```

### [Pistonball](https://pettingzoo.farama.org/environments/butterfly/pistonball/)

The default parameters are just for demonstration purposes. If you want to train a model that can complete the task, you can try the parameters below:

```bash
python tianshou-pistonball-discrete.py --piston-num 5 --num-epochs 2000 --buffer-size 100000 --step-per-epoch 1000 --batch-size 128
```

It requires about 32GB of memory and 1500 MB of video memory to train a model with the above settings.

Use the following command to watch the game with a trained model:

```bash
# load a trained model and watch the game without training
python tianshou-pistonball-discrete.py --piston-num 5 --watch-only --watch-episode-num 5 --model-path ./path/to/trained/model.pth

# this repository provides a trained model that may be able to perform the task :)
python tianshou-pistonball-discrete.py --piston-num 5 --watch-only --watch-episode-num 5 --model-path ./models/pistonball-discrete/best.pth --seed 1
```
