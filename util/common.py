import random
import importlib
import torch
from collections import OrderedDict
from pathlib import Path

def mkdir(dir_path, delete=False, parents=True):
    import shutil
    if not isinstance(dir_path, Path):
        dir_path = Path(dir_path)
    if delete:
        if dir_path.exists():
            shutil.rmtree(str(dir_path))
    if not dir_path.exists():
        dir_path.mkdir(parents=parents)

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def get_filenames(dir_path, exts=['png', 'jpg'], recursive=True):
    '''
    Get the file paths in the given folder.
    param exts: list, e.g., ['png',]
    return: list
    '''
    if not isinstance(dir_path, Path):
        dir_path = Path(dir_path)

    file_paths = []
    for current_ext in exts:
        if recursive:
            file_paths.extend([str(x) for x in dir_path.glob('**/*.'+current_ext)])
        else:
            file_paths.extend([str(x) for x in dir_path.glob('*.'+current_ext)])

    return file_paths

def readline_txt(txt_file):
    txt_file = [txt_file, ] if isinstance(txt_file, str) else txt_file
    out = []
    for txt_file_current in txt_file:
        with open(txt_file_current, 'r') as ff:
            out.extend([x[:-1] for x in ff.readlines()])

    return out

def scan_files_from_folder(dir_paths, exts, recursive=True):
    '''
    Scaning images from given folder.
    Input:
        dir_pathas: str or list.
        exts: list
    '''
    exts = [exts, ] if isinstance(exts, str) else exts
    dir_paths = [dir_paths, ] if isinstance(dir_paths, str) else dir_paths

    file_paths = []
    for current_dir in dir_paths:
        for current_ext in exts:
            if recursive:
                search_flag = f"**/*.{current_ext}"
            else:
                search_flag = f"*.{current_ext}"
            file_paths.extend(sorted([str(x) for x in Path(current_dir).glob(search_flag)]))

    return file_paths

def write_path_to_txt(dir_folder, txt_path, search_key, num_files=None):
    '''
    Scaning the files in the given folder and write them into a txt file
    Input:
        dir_folder: path of the target folder
        txt_path: path to save the txt file
        search_key: e.g., '*.png'
    '''
    txt_path = Path(txt_path) if not isinstance(txt_path, Path) else txt_path
    dir_folder = Path(dir_folder) if not isinstance(dir_folder, Path) else dir_folder
    if txt_path.exists():
        txt_path.unlink()
    path_list = [str(x) for x in dir_folder.glob(search_key)]
    random.shuffle(path_list)
    if num_files is not None:
        path_list = path_list[:num_files]
    with open(txt_path, mode='w') as ff:
        for line in path_list:
            ff.write(line+'\n')

def reload_model(model, ckpt):
    if list(model.state_dict().keys())[0].startswith('module.'):
        if list(ckpt.keys())[0].startswith('module.'):
            ckpt = ckpt
        else:
            ckpt = OrderedDict({f'module.{key}':value for key, value in ckpt.items()})
    else:
        if list(ckpt.keys())[0].startswith('module.'):
            ckpt = OrderedDict({key[7:]:value for key, value in ckpt.items()})
        else:
            ckpt = ckpt
    model.load_state_dict(ckpt)

def load_model(model, ckpt_path=None, device='cuda:0'):
    state = torch.load(ckpt_path, map_location=device)
    print(state.keys())
    if 'state_dict' in state:
        state = state['state_dict']
    if 'model' in state:
        state = state['model']
    reload_model(model, state)

def decode_first_stage(sample_out, first_stage_model=None):
        ori_dtype = sample_out.dtype
        if first_stage_model is None:
            return sample_out
        else:
              sample_out = 1 / 4 * sample_out
              sample_out = sample_out.type(next(first_stage_model.parameters()).dtype)
 
              out = first_stage_model.decode(sample_out)
              return out.type(ori_dtype)

def encode_first_stage(x_start, first_stage_model=None):
        ori_dtype = x_start.dtype
        if first_stage_model is None:
            return x_start
        else:
            with torch.no_grad():
                x_start = x_start.type(dtype=next(first_stage_model.parameters()).dtype)
                x_start = first_stage_model.encode(x_start)[0]
                out = x_start * 4
                return out.type(ori_dtype)

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
