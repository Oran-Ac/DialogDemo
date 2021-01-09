from dataset_init import *
from models import *

api_args = {
    'token': 'xxxxx',
    'mongodb': {
        'database': 'dialog',
        'table': 'test',
    },
    'special_cmd': {
        'clear': '#clear',
        'kg': '#kg',
    },
    'host': '0.0.0.0',
    'port': 8080,
    'model': 'bertretrieval',
    'chat_mode': 0,
    'multi_turn_size': 5,
    'verbose': True,
    'gpu_id': 0,
}

dataset_loader = {
    'PostTraing':load_data_for_post_training,
}

agent_map={
    'PostTraing':PostTrainingAgent,
}

model_parameters = {
    'PostTraing': [('multi_gpu', 'total_steps'), {'run_mode': 'mode', 'local_rank': 'local_rank', 'model': 'bimodel', 'lang': 'lang'}],
}

def collect_parameter_4_model(args):
    if args['model'] in model_parameters:
        parameter_map, parameter_key = model_parameters[args['model']]
        parameter_map = [args[key] for key in parameter_map]
        parameter_key = {key: args[value] for key, value in parameter_key.items()}
        return parameter_map, parameter_key
    else:
        raise Exception(f'[!] cannot find the model {args["model"]}')
        
def load_dataset(args):
    if args['model'] in dataset_loader:
        return dataset_loader[args['model']](args)
    else:
        raise Exception(f'[!] cannot find the model {args["model"]}')