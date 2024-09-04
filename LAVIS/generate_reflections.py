from typing import List, Dict, Any

from emma_utils import *

with open("./prompts/reflexion_few_shot_examples.txt", 'r') as f:
    FEW_SHOT_EXAMPLES = f.read()

def _get_scenario(s: str) -> str:
    """Parses the relevant scenario from the experience log."""
    return s.split("\n\n")[-2].strip()

def _generate_reflection_query(log_str: str, memory: List[str]) -> str:
    """Allows the Agent to reflect upon a past experience."""
    scenario: str = _get_scenario(log_str)
    query: str = f"""You will be given the history of a past experience in which you were placed in an environment and given a task to complete. You were unsuccessful in completing the task. Do not summarize your environment, but rather think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken. For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. You will need this later when you are solving the same task. Give your plan after "Plan". Here are two examples:

{FEW_SHOT_EXAMPLES}
{scenario}\n"""

    if len(memory) > 0:
        query += '\nPlans from past attempts:\n'
        for i, m in enumerate(memory):
            query += f'Trial #{i}: {m}\n'

    query += 'New plan: '
    return query

def update_memory(trial_log_path: str, env_configs: Dict[str, Dict[str, Any]], model: str = 'gpt-35-turbo-16k') -> Dict[str, Dict[str, Any]]:
    """Updates the given env_config with the appropriate reflections."""
    with open(trial_log_path, 'r') as f:
        full_log: str = f.read()
        
    env_logs: List[str] = full_log.split('#####\n\n#####')
    assert len(env_logs) == len(env_configs), print(f'bad: {len(env_logs)}, {len(env_configs)}')

    for i, env in env_configs.items():
        # if unsolved, get reflection and update env config
        if not env['is_success']:
            if len(env['memory']) > 3:
                memory: List[str] = env['memory'][-3:]
            else:
                memory: List[str] = env['memory']

            reflection_query: str = _generate_reflection_query(env_logs[int(i.split("_")[-1])], memory)
            
            if model == "text-davinci-003":
                reflection: str = get_completion(reflection_query, model)
            else:
                reflection: str = get_chat(reflection_query, model)
                          
            env_configs[i]['memory'] += [reflection]
                
    return env_configs
