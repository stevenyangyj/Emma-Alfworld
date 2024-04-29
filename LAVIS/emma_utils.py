from abc import ABC, abstractmethod
from typing import Union, List, Dict, Optional
import os
import gc
import sys
import math
import time
import logging

import h5py
import numpy as np
import torch as th
import openai
import tiktoken
# import eventlet
from tenacity import (
    retry,
    stop_after_attempt, # type: ignore
    wait_random_exponential,
)

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
    
    
def release_memory():
    th.cuda.empty_cache()
    _ = gc.collect()


Model = Literal["gpt-4", "gpt-35-turbo", "gpt-35-turbo-16k", "text-davinci-003"]

TOKEN_LIMIT = {
    "gpt-35-turbo-16k": 16384,
    "gpt-35-turbo": 4096,
    "text-davinci-003": 4097,
}

os.environ["OPENAI_API_KEY"] = "9113c4ea-0fbf-4414-8c2f-61445917d294"
os.environ["OPENAI_API_BASE"] = "http://gpt-proxy.jd.com/gateway/azure"

openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_base = os.getenv('OPENAI_API_BASE')

rate_limit_per_second = 10
delay = 1.0 / rate_limit_per_second


def num_tokens_from_messages(messages: dict):
    encoding= tiktoken.get_encoding("cl100k_base")  #model to encoding mapping https://github.com/openai/tiktoken/blob/main/tiktoken/model.py
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens


def num_tokens_from_string(prompts: str):
    encoding= tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(prompts))


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(20))
def get_completion(prompt: str, model: str = 'text-davinci-003', temperature: float = 0.0, max_tokens: int = 256, stop_strs: Optional[List[str]] = None) -> str:
    assert model == "text-davinci-003"

    token_num = num_tokens_from_string(prompt)
    while token_num + max_tokens > TOKEN_LIMIT[model]:
        print("******allowed tokens exceeding******")
        print(f"Tokens#: {token_num + max_tokens}")
        prompt = prompt[-int((TOKEN_LIMIT[model] - max_tokens) * 2.5):]
        token_num = num_tokens_from_string(prompt)
        print(f"Tokens# after clipping: {token_num + max_tokens}")

    # Sleep for the delay
    time.sleep(delay)

    # eventlet.monkey_patch()
    # with eventlet.Timeout(timeout_threshold, False):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop_strs,
    )
    
    # print('Response: ', response)
    try:
        # normal response
        text = response["choices"][0]["text"].split("\n")[0]
    except KeyError:
        if response["error"]["code"] == "429" or response["error"]["code"] == "9999":
            # trigger error code
            logging.warning(f'trigger error code: {response["error"]["code"]}')
            raise Exception("Error Code "+ response["error"]["code"])
        else:
            # trigger content filtering
            logging.warning(f'may trigger content filtering: {response["error"]["code"]}')
            text = "9999999999"
    # print('Action: ', text)
    return text


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(20))
def get_chat(prompt: str, model: Model, temperature: float = 0.0, max_tokens: int = 256, stop_strs: Optional[List[str]] = None) -> str:
    assert model != "text-davinci-003"
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    # counting prompt tokens + maximal response tokens
    conv_history_tokens = num_tokens_from_messages(messages)
    while conv_history_tokens + max_tokens >= TOKEN_LIMIT[model]:
        print("******allowed tokens exceeding******")
        print(messages)
        print(f"Tokens#: {conv_history_tokens + max_tokens}")
        prompt = prompt[-int((TOKEN_LIMIT[model]-max_tokens-7)*3.0):]
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        conv_history_tokens = num_tokens_from_messages(messages)
        print(f"Tokens# after clipping: {conv_history_tokens + max_tokens}")

    # Sleep for the delay
    time.sleep(delay)

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        stop=stop_strs,
        temperature=temperature,
        stream=False,
    )
    
    # print('Response: ', response)
    try:
        text = response["choices"][0]["message"]["content"]
    except KeyError:
        try:
            if response["error"]["code"] is not None:
                raise Exception("Error Code "+ response["error"]["code"])
        except KeyError:
            text = ""
    return text


def llm_forward(prompt: str, model: Model, stop: List[str] = ["\n"]):
    text = ""
    cur_try = 0

    while len(text.strip()) < 5 and cur_try < 6:
        text = get_completion(prompt=prompt, model=model, temperature=cur_try * 0.2, max_tokens=128, stop_strs=stop)
        cur_try += 1

    return text


class EnvironmentHistory:
    def __init__(self, base_query: str, start_info, memory: List[str], history: List[Dict[str, str]] = []) -> None:
        self._cur_query: str = f'{_get_base_query(base_query, start_info, memory)}'
        self._history: List[Dict[str, str]] = history
        self._last_action: str = ''
        self._is_exhausted: bool = False

    def add(self, label: str, value: str) -> None:
        assert label in ['action', 'observation', 'human_edit']
        self._history += [{
            'label': label,
            'value': value,
        }]
        if label == 'action':
            if value == self._last_action:
                self._is_exhausted = True
            else:
                self._last_action = value

    def check_is_exhausted(self) -> bool:
        return self._is_exhausted

    def reset(self) -> None:
        self._history = []

    def __str__(self) -> str:
        s: str = self._cur_query + '\n'
        for i, item in enumerate(self._history):
            if item['label'] == 'action':
                s += f'> {item["value"]}'
            elif item['label'] == 'observation':
                s += item['value']
            # NOT CURRENTLY SUPPORTED
            elif item['label'] == 'human_edit':
                s += f'[human edit]: {item["value"]}'
            s += '\n'
        return s

def _get_base_query(base_query: str, start_info: str, memory: List[str]) -> str:
    query = base_query

    # add memory if it exists
    if len(memory) > 0:
        query += '\nYour memory for the task below:'
        for i, m in enumerate(memory):
            query += f'\nTrial {i}: {m.strip()}'
        query += '\n'
    query += f"\n{start_info}."
    return query

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        device: Union[th.device, str] = "auto",
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.visual_obs_shape = (3, 224, 224)

        self.pos = 0
        self.full = False
        self.device = device

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds)

    @abstractmethod
    def _get_samples(
        self, batch_inds: np.ndarray):
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()
    
    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return th.tensor(array, dtype=th.float16, device=self.device)
        return th.as_tensor(array, dtype=th.float16, device=self.device)
    

class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        device: Union[th.device, str] = "auto",
    ):
        super().__init__(buffer_size, device)
        # Adjust buffer size
        self.buffer_size = max(buffer_size, 1)

        self.visual_obs = np.zeros((buffer_size, *self.visual_obs_shape))
        self.text_inputs = [None for _ in range(buffer_size)]
        self.text_outputs = [None for _ in range(buffer_size)]

    def add(
        self,
        visual_ob: np.ndarray,
        text_input: str,
        text_output: str,
    ) -> None:

        self.visual_obs[self.pos] = visual_ob
        self.text_inputs[self.pos] = text_input
        self.text_outputs[self.pos] = text_output

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int):
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        return super().sample(batch_size=batch_size)

    def _get_samples(self, batch_inds: np.ndarray):

        data = {
            "image": self.to_torch(self.visual_obs[batch_inds, :]),
            "text_input": [self.text_inputs[i] for i in batch_inds],
            "text_output": [self.text_outputs[i] for i in batch_inds],
        }
        return data
    
    
class DPOReplayBuffer(BaseBuffer):
    """
    DPO's replay buffer
    """

    def __init__(
        self,
        buffer_size: int,
        device: Union[th.device, str] = "auto",
    ):
        super().__init__(buffer_size, device)
        # Adjust buffer size
        self.buffer_size = max(buffer_size, 1)

        self.visual_obs = np.zeros((buffer_size, *self.visual_obs_shape))
        self.text_inputs = [None for _ in range(buffer_size)]
        self.w_text_outputs = [None for _ in range(buffer_size)]
        self.l_text_outputs = [None for _ in range(buffer_size)]

    def add(
        self,
        visual_ob: np.ndarray,
        text_input: str,
        w_text_output: str,
        l_text_output: str
    ) -> None:

        self.visual_obs[self.pos] = visual_ob
        self.text_inputs[self.pos] = text_input
        self.w_text_outputs[self.pos] = w_text_output
        self.l_text_outputs[self.pos] = l_text_output

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int):
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        return super().sample(batch_size=batch_size)

    def _get_samples(self, batch_inds: np.ndarray):

        data = {
            "image": self.to_torch(self.visual_obs[batch_inds, :]),
            "text_input": [self.text_inputs[i] for i in batch_inds],
            "text_output_pos": [self.w_text_outputs[i] for i in batch_inds],
            "text_output_neg": [self.l_text_outputs[i] for i in batch_inds]
        }
        return data
    

def save_checkpoint(model, optimizer, scaler, cur_epoch, output_dir: str=""):
    """
    Save the checkpoint at the current epoch.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model_no_ddp = model
    param_grad_dic = {
        k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
    }
    state_dict = model_no_ddp.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            # delete parameters that do not require gradient
            del state_dict[k]

    save_obj = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": cur_epoch,
    }
    save_to = os.path.join(
        output_dir,
        "emma_checkpoint_{}.pth".format(cur_epoch),
    )
    print("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
    th.save(save_obj, save_to)

def load_checkpoint(model, optimizer, scaler, path_checkpoint, device):
    """
    Resume from a checkpoint.
    """
    if os.path.isfile(path_checkpoint):
        checkpoint = th.load(path_checkpoint, map_location=device)
    else:
        raise RuntimeError("checkpoint path is invalid")

    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict, strict=False)

    optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    cur_epoch = checkpoint["epoch"] + 1
    print("Resume checkpoint from {}".format(path_checkpoint))

    return model, optimizer, scaler, cur_epoch