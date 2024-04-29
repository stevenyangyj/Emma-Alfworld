import hashlib
import http.server
import socketserver
import threading
import queue
import time
import logging
import json
import subprocess
import os
import re
import gc
import ast
import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn
from PIL import Image

from dataclasses import dataclass
from typing import Dict, Any

from lavis.models import load_model_and_preprocess
from lavis.common.optims import LinearWarmupCosineLRScheduler
from emma_utils import *
from generate_reflections import update_memory
from post_processing import action_postprocess, process_ob

# create action and feedback queue
action_queue = queue.Queue()
feedback_queue = queue.Queue()

@dataclass
class FeedbackData:
    history: str = ''
    observation: str = ''
    task_type: str = ''
    information: Dict[str, Any]  = ''
    done: bool = False
    image: np.ndarray = None

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  cudnn.benchmark = False
  cudnn.deterministic = True

# http server thread
class ServerThread(threading.Thread):
    def __init__(self, host, port, action_queue, feedback_queue):
        super(ServerThread, self).__init__()
        self.host = host
        self.port = port
        self.action_queue = action_queue
        self.feedback_queue = feedback_queue
        
    def run(self):
        class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass
            def do_POST(self):
                try:
                    # get request
                    content_length = int(self.headers['Content-Length'])
                    data = self.rfile.read(content_length)

                    # check data integrity
                    received_md5 = self.headers.get('MD5')
                    calculated_md5 = hashlib.md5(data).hexdigest()
                    if received_md5 != calculated_md5:
                        self.send_response(400)
                        self.end_headers()
                    else:
                        def feedback_queue_put(feedback):
                            feedback_queue.put(feedback)

                        # put feedback into feedback_queue
                        # print("put feedback to queue")
                        feedback_queue_put(data.decode())

                        # wait for feedback
                        action = action_queue.get()
                        # print("get next action from queue")
                        
                        # send response
                        response_data = str(action)
                        response_data = response_data.encode()
                        response_md5 = hashlib.md5(response_data).hexdigest()

                        self.send_response(200)
                        self.send_header('MD5', response_md5)
                        self.end_headers()
                        self.wfile.write(response_data)
                except Exception as e:
                    self.send_error(400, str(e))

        with socketserver.TCPServer((self.host, self.port), MyHTTPRequestHandler) as httpd:
            print("Server running on IP:", self.host, "PORT:", self.port)
            httpd.serve_forever()

 # 调用命令kill之前的监听进程
def release_port():
    command = "ss -lptn 'sport = :7860'"
    output = subprocess.check_output(command, shell=True).decode()
    # 解析输出结果获取PID
    lines = output.strip().split('\n')
    if len(lines) > 1:
        # 忽略标题行，获取第一行数据
        line = lines[1]
        # 使用正则表达式提取PID
        pid_pattern =  r',pid=(\d+),'
        match =re.search(pid_pattern, line)
        if match:
            pid = match.group(1)
            # 杀死对应的进程
            os.kill(int(pid), 9)
            time.sleep(5)

def process_feedback(text):
    # 使用正则表达式匹配并提取字段
    # pattern = r'\[#OBSERVATION\](.*?)\[#IMAGE\](.*)'
    pattern = r'\[#OBSERVATION\](.*?)\[#HISTORY\](.*?)\[#INFORMATION\](.*?)\[#TYPE\](.*?)\[#DONE\](.*?)\[#IMAGE\](.*)'
    mat = re.match(pattern, text, re.DOTALL)
    feedback_data = FeedbackData()
    # 将匹配到的字段存储到相应的变量中
    if mat:
        feedback_data.observation = mat.group(1)
        
        feedback_data.history = mat.group(2)
        # 将info由str转为dict
        feedback_data.information = ast.literal_eval(mat.group(3))

        feedback_data.task_type = mat.group(4)
        
        if mat.group(5) == "True":
            feedback_data.done = True
        else:
            feedback_data.done = False
        
        # 将接收到的数据解析为Python对象
        py_data = json.loads(mat.group(6))
        # 将Python对象转换为numpy数组并设置dtype为uint8
        feedback_data.image = Image.fromarray(np.asarray(py_data, dtype=np.uint8))
        # cv2.imwrite('1.jpg', image)
        # feedback_data.image.save('1.jpg')

    return feedback_data

release_port()
PORT=7860
IP="0.0.0.0" # 自动获取本地IP
server_thread = ServerThread(IP, PORT, action_queue, feedback_queue)
server_thread.start()

# random seeds
seed = 42
set_seed(seed)
# experiment configs
num_rounds = 12
num_envs = 134
run_training = True
save_ckpt = True
now = int(time.time())
time_array = time.localtime(now)
format_time = time.strftime("%Y%m%d%H%M%S", time_array)
method = "dagger_server_human_desc"
enable_dpo = True
enable_tc = False
output_dir = f"/home/yangyijun14/repos/EMMA/LAVIS/align_outputs/{method}/with_bc_dpo-{enable_dpo}-tc-{enable_tc}-{format_time}/"

relexion_dir = os.path.join(output_dir, 'logging_results')
if not os.path.exists(relexion_dir):
    os.makedirs(relexion_dir)

logging.basicConfig(filename=os.path.join(output_dir, f'running_nb01.log'), encoding='utf-8', level=logging.INFO)
logger = logging.getLogger("dagger_server_running")
logger.info(f'{"Dagger Procedure Start Running":*^40}')

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# loads EMMA model
model, vis_processors, _ = load_model_and_preprocess(name="blip2_emma", model_type="vicuna7b", is_eval=False, device=device)

# reflexion configs
PREFIXES = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}
FOLDER = './prompts'
PROMPT_FILE = 'alfworld_3prompts.json'
with open(os.path.join(FOLDER, PROMPT_FILE), 'r') as f:
    d = json.load(f)

env_configs: Dict[str, Dict[str, Any]] = dict()
for i in range(num_envs):
    env_configs[f'env_{i}'] = {
        'memory': [],
        'is_success': False,
        'type': None,
        'path_length': 0
    }
    
if enable_dpo:
    buffer = DPOReplayBuffer(buffer_size=int(1e5), device=device)
else:
    buffer = ReplayBuffer(buffer_size=int(1e5), device=device)

train_iters_per_epoch = 5
if enable_dpo:
    accum_grad_steps = 4
    batch_size = 4
else:
    accum_grad_steps = 4
    batch_size = 8
lr_scale = 1
weight_decay = 0.05
optim_params = model.get_optimizer_params(weight_decay, lr_scale)
num_parameters = 0
for p_group in optim_params:
    for p in p_group["params"]:
        num_parameters += p.data.nelement()    
logger.info("number of trainable parameters: {}".format(num_parameters))

init_lr = 1e-5
warmup_lr = 1e-8
warmup_steps = 300
beta2 = 0.999
optimizer = torch.optim.AdamW(
    optim_params,
    lr=init_lr,
    betas=(0.9, beta2),
)
    
lr_sched_cls = LinearWarmupCosineLRScheduler
lr_scheduler = lr_sched_cls(
    optimizer=optimizer,
    max_epoch=num_rounds,
    min_lr=0,
    init_lr=init_lr,
    decay_rate=None,
    warmup_start_lr=warmup_lr,
    warmup_steps=warmup_steps,
)

# use pytorch amp
scaler = torch.cuda.amp.GradScaler()

# initialize while loop
# Wait for requestion
data = feedback_queue.get()
feedback_data = process_feedback(data)
# reset environment prompt
for i, (k, v) in enumerate(PREFIXES.items()):
    if feedback_data.task_type.startswith(k):
        iter = 0
        env_configs['env_0']['type'] = feedback_data.task_type
        if not env_configs['env_0']['is_success']:
            text_inputs = []
            text_outputs_llm = []
            text_outputs_vlm = []
            image_inputs = []
            if len(env_configs['env_0']['memory']) > 0:
                base_prompt = d[f'react_{v}_1'] + "\n" + d[f'react_{v}_2']
            else:
                base_prompt = d[f'react_{v}_0'] + "\n" + d[f'react_{v}_1'] + "\n" + d[f'react_{v}_2']
            init_ob = '\n'.join(feedback_data.observation.split('\n\n')[1:])
            env_history = EnvironmentHistory(base_prompt, init_ob, env_configs['env_0']['memory'], [])

# easy to ignore
model.eval()
trial_idx = 0
world_log_path: str = os.path.join(relexion_dir, f'world.log')
while trial_idx < num_rounds:
    trial_log_path: str = os.path.join(relexion_dir, f'trial_{trial_idx}.log')
    trial_env_configs_log_path: str = os.path.join(relexion_dir, f'env_results_trial_{trial_idx}.json')
    cur_task = 0
    success = 0
    additional_success = 0
    training_step = 0
    start_time = time.time()

    while cur_task < num_envs:
        if env_configs[f'env_{cur_task}']["is_success"]:
            success += 1
            action_queue.put(["SKIP"])
        else:
            if feedback_data.task_type == "":
                if vlm_action.startswith("think:"):
                    env_history.add("observation", "OK.")
                else:
                    env_history.add("observation", process_ob(feedback_data.observation))

            image = vis_processors["eval"](feedback_data.image).unsqueeze(0).to(device)

            # llm forward
            llm_action = llm_forward(str(env_history) + "> ", stop=['\n'], model="text-davinci-003").strip()

            # not imitating thought trace
            while llm_action.startswith("think:") and (not enable_tc):
                env_history.add("action", llm_action)
                env_history.add("observation", "OK.")
                llm_action = llm_forward(str(env_history) + "> ", stop=['\n'], model="text-davinci-003").strip()

            # vlm forward
            vlm_action = model.generate({"image": image, "prompt": feedback_data.history}, 
                                        use_nucleus_sampling=False, 
                                        max_length=128, 
                                        min_length=5, 
                                        num_beams=5, 
                                        top_p=1.0,
                                        repetition_penalty=1.0,
                                        length_penalty=0.0
                                        )[0]

            if llm_action.startswith("think:"):
                pass
            else:
                if llm_action.startswith("9"):
                    if vlm_action.startswith("think:"):
                        llm_action = vlm_action
                    else:
                        llm_action = action_postprocess(vlm_action)
                else:
                    llm_action = action_postprocess(llm_action)

            if vlm_action.startswith("think:"):
                pass
            else:
                vlm_action = action_postprocess(vlm_action)

            env_history.add("action", vlm_action)

            logger.info(f"Task: [{cur_task}], Iter: [{iter}], Obs: {process_ob(feedback_data.observation)}, VLM Action: {vlm_action}, LLM Action: {llm_action}")

            iter += 1
            action_queue.put([vlm_action])

            if run_training:
                text_inputs.append(feedback_data.history)
                text_outputs_llm.append(llm_action)
                text_outputs_vlm.append(vlm_action)
                image_inputs.append(image.squeeze(0).cpu().numpy())

            if feedback_data.done:
                env_configs[f'env_{cur_task}']['path_length'] = iter
                env_configs[f'env_{cur_task}']['is_success'] = True
                success += 1
                additional_success += 1

        # Wait for request
        data = feedback_queue.get()
        feedback_data = process_feedback(data)
        # reset environment prompt
        for i, (k, v) in enumerate(PREFIXES.items()):
            if feedback_data.task_type.startswith(k):
                release_memory()

                if run_training and not env_configs[f'env_{cur_task}']["is_success"]:
                    # store all expert actions
                    for j in range(len(text_inputs)):
                        if enable_dpo:
                            buffer.add(
                                visual_ob=image_inputs[j],
                                text_input=text_inputs[j],
                                w_text_output=text_outputs_llm[j],
                                l_text_output=text_outputs_vlm[j]
                            )
                        else:
                            buffer.add(
                                visual_ob=image_inputs[j],
                                text_input=text_inputs[j],
                                text_output=text_outputs_llm[j]
                            )

                    # then finetune vlm to align with llm
                    logger.info(f'{"Start Training":*^40}')
                    model.train()
                    for e in range(train_iters_per_epoch * accum_grad_steps):
                        samples = buffer.sample(batch_size)

                        lr_scheduler.step(cur_epoch=trial_idx, cur_step=training_step)

                        with torch.cuda.amp.autocast(enabled=True):
                            if enable_dpo:
                                loss_metrics = model.dpo_forward(samples)
                                loss = loss_metrics["loss"]
                                w_reward = loss_metrics["chosen_reward"]
                                l_reward = loss_metrics["rejected_reward"]
                                w_entropy = loss_metrics["chosen_entropy"]
                                logger.info(f'Epoch: [{trial_idx}], Iter: [{training_step}], Lr: [{optimizer.param_groups[0]["lr"]:.6f}], Loss: [{loss.item():.4f}], W_reward: [{w_reward.item():.4f}], L_reward: [{l_reward.item():.4f}], NLL: [{w_entropy.item():.4f}]')
                            else:
                                loss_metrics = model.forward(samples)
                                loss = loss_metrics["loss"]
                                logger.info(f'Epoch: [{trial_idx}], Iter: [{training_step}], Lr: [{optimizer.param_groups[0]["lr"]:.6f}], Loss: [{loss.item():.4f}]')

                        scaler.scale(loss / accum_grad_steps).backward()

                        if (e + 1) % accum_grad_steps == 0:
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            release_memory()

                        training_step += 1

                    # easy to ignore
                    model.eval()

                if env_configs[f'env_{cur_task}']["is_success"]:
                    with open(trial_log_path, 'a') as wf:
                        wf.write(f'\n#####\n\nEnvironment #{cur_task}: Success\n\n#####\n')
                else:
                    with open(trial_log_path, 'a') as wf:
                        wf.write(f'\n#####\n\nEnvironment #{cur_task}:\n{str(env_history)}STATUS: {"FAIL"}\n\n#####\n')

                iter = 0
                cur_task += 1
                if cur_task < num_envs:
                    env_configs[f'env_{cur_task}']['type'] = feedback_data.task_type
                    if not env_configs[f'env_{cur_task}']['is_success']:
                        text_inputs = []
                        text_outputs_llm = []
                        text_outputs_vlm = []
                        image_inputs = []
                        if len(env_configs[f'env_{cur_task}']['memory']) > 0:
                            base_prompt = d[f'react_{v}_1'] + "\n" + d[f'react_{v}_2']
                        else:
                            base_prompt = d[f'react_{v}_0'] + "\n" + d[f'react_{v}_1'] + "\n" + d[f'react_{v}_2']
                        init_ob = '\n'.join(feedback_data.observation.split('\n\n')[1:])
                        if len(env_configs[f'env_{cur_task}']['memory']) > 3:
                            env_history = EnvironmentHistory(base_prompt, init_ob, env_configs[f'env_{cur_task}']['memory'][-3:], [])
                        else:
                            env_history = EnvironmentHistory(base_prompt, init_ob, env_configs[f'env_{cur_task}']['memory'], [])

    env_configs = update_memory(trial_log_path, env_configs, model='text-davinci-003')

    for i, (k, v) in enumerate(PREFIXES.items()):
        if feedback_data.task_type.startswith(k) and not env_configs['env_0']['is_success']:
            text_inputs = []
            text_outputs_llm = []
            text_outputs_vlm = []
            image_inputs = []
            if len(env_configs['env_0']['memory']) > 0:
                base_prompt = d[f'react_{v}_1'] + "\n" + d[f'react_{v}_2']
            else:
                base_prompt = d[f'react_{v}_0'] + "\n" + d[f'react_{v}_1'] + "\n" + d[f'react_{v}_2']
            init_ob = '\n'.join(feedback_data.observation.split('\n\n')[1:])
            if len(env_configs['env_0']['memory']) > 3:
                env_history = EnvironmentHistory(base_prompt, init_ob, env_configs['env_0']['memory'][-3:], [])
            else:
                env_history = EnvironmentHistory(base_prompt, init_ob, env_configs['env_0']['memory'], [])

    log_str: str = f"""
-----
ROUND: {trial_idx}
SUCCESS: {success}
ADDITIONAL SUCCESS: {additional_success}
FAIL: {num_envs - success}
TOTAL: {num_envs}
ACCURACY: {round(success / num_envs, 2)}
TIME: {time.time() - start_time:.2f} s
-----\n\n"""
    with open(world_log_path, 'a') as wf:
        wf.write(log_str)

    with open(trial_env_configs_log_path, 'w') as wf:
        json.dump(env_configs, wf, indent=4)

    if save_ckpt:
        # save emma model
        save_checkpoint(model, optimizer, scaler, trial_idx, output_dir)

    trial_idx += 1
    
    
release_port()