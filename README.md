# Single Agent CartPole PPO with Ray RLlib

This project uses Ray RLlib's Proximal Policy Optimization (PPO) algorithm to train an agent to play the `CartPole-v1` environment from Gymnasium (formerly OpenAI Gym). The script includes options for training a new model, loading a pre-trained model, and recording video of the agent's performance.

<br>

## Project Structure

- **`single_agent_cartpole.py`**: The main script that trains the PPO agent on the CartPole-v1 environment, saves or loads models, and records evaluation episodes.
- **`requirements.txt`**: Contains the list of dependencies required for this project.
- **`venv/`**: (Optional) A virtual environment to run the project in an isolated environment.
- **`saved_model/`**: Directory where trained models are saved (created during execution).
- **`video/`**: Directory where evaluation videos are saved (created during execution).

<br>

## Prerequisites

- Python 3.9 or higher (3.11 recommended)
- `ffmpeg` (to enable video recording in Gymnasium environments)
- Virtual environment (optional but recommended)

<br>
<br>

## Setup Instructions

### 1. Clone the repository

```
git clone https://github.com/shannenlolol/SingleAgentCartpoleRLlib.git
cd single_agent_cartpole
```
<br>

### 2. Create and activate a virtual environment

It is recommended to use a virtual environment to keep the dependencies isolated.

#### On macOS/Linux:
```
python3 -m venv venv
source venv/bin/activate
```

#### On Windows:
```
python -m venv venv
venv\Scripts\activate
```

<br>

### 3. Install dependencies

With the virtual environment activated, install the required dependencies from `requirements.txt`:

```
pip install -r requirements.txt
```

<br>

### 4. Install `ffmpeg`

To enable video recording of evaluation episodes, you need to install `ffmpeg`. Follow the instructions below based on your operating system.

#### On macOS (with Homebrew):
```
brew install ffmpeg
```

#### On Linux (with APT):
```
sudo apt update
sudo apt install ffmpeg
```

#### On Windows:
Download and install `ffmpeg` from the official site: https://ffmpeg.org/download.html.

Make sure `ffmpeg` is added to your system’s PATH so it can be accessed from the command line.


<br>
<br>

## How to Run the Project

The `single_agent_cartpole.py` script allows you to either train a new model or load a pre-trained model for evaluation. You can also record videos of the agent's performance.

### 1. Train a New Model

By default, running the script without any arguments will train a new PPO model on the `CartPole-v1` environment.

```
python single_agent_cartpole.py
```

<br>

### 2. Load a Pre-Trained Model

You can load a previously saved model using the `--load` flag and the `--model_path` argument:

```
python single_agent_cartpole.py --load --model_path ./saved_model/20240828_105147
```

<br>

This will skip training and directly evaluate the loaded model.


<br>

### 3. Specify the Number of Evaluation Episodes

You can also control the number of evaluation episodes by using the `--num_episodes` argument:

```
python single_agent_cartpole.py --num_episodes 10
```

<br><br>

### Full Example:

To load a pre-trained model and run 5 evaluation episodes:

```
python single_agent_cartpole.py --load --model_path ./saved_model/20240828_105147 --num_episodes 5
```


<br><br>

## Script Arguments

| Argument        | Description                                                                                          | Example                                           |
|-----------------|------------------------------------------------------------------------------------------------------|---------------------------------------------------|
| `--load`        | Load a pre-trained model (if provided, skips training and directly evaluates the model).              | `--load`                                          |
| `--model_path`  | Path to the model to load or save.                                                                    | `--model_path ./saved_model/20240828_105147`      |
| `--num_episodes`| Specify the number of episodes to evaluate the agent's performance. Default is 4 episodes.            | `--num_episodes 10`                               |

<br>

## Output

- **Model Saving**: The model is saved in the `./saved_model/` directory with a timestamp if training is completed.
- **Video Recording**: Evaluation episodes are recorded and saved in the `./video/` directory as `.mp4` files.
