from __future__ import print_function
import datetime
import os
import shutil
import yaml
import time
import sys
import numpy as np
import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F
import math
import utils
import argparse
import importlib

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "config.yaml")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_args_and_set_gpu():
    parser = argparse.ArgumentParser(description="PePNet training")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA device ID, for example 0 or 0,1")
    parser.add_argument("--config", default=DEFAULT_CONFIG, type=str, help="path to the YAML config")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    return args

class Logger(object):
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(config_dict)


def validate_data_paths(cfg):
    for key in ("path", "pose_path"):
        value = getattr(cfg.data, key, "")
        if not value:
            raise ValueError(
                f"data.{key} is empty; set it in config.yaml before training"
            )
        if not os.path.isdir(value):
            raise FileNotFoundError(
                f"data.{key} directory does not exist: {value}"
            )

def train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, print_freq, tau_value, use_pose_in_model):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('clips/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))

    header = f'Epoch: [{epoch}]'

    for data, target, _ in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        clip = data['clip'].to(device)
        pose = data['pose'].to(device) if 'pose' in data and use_pose_in_model else None
        target = target.to(device)

        if use_pose_in_model: # Check if model should use pose (based on config)
            output = model(clip, pose, tau_value)
        else:
            output = model(clip) # Pass tau even if pose is not used, if model expects it
                                                # Or adjust model signature if tau is only with pose
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = clip.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['clips/s'].update(batch_size / (time.time() - start_time))
        sys.stdout.flush()
    lr_scheduler.step()

def evaluate(model, criterion, data_loader, device, tau_value, use_pose_in_model):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    video_prob = {}
    video_label = {}
    with torch.no_grad():
        for data, target, video_idx in metric_logger.log_every(data_loader, 100, header):
            clip = data['clip'].to(device, non_blocking=True)
            pose = data['pose'].to(device, non_blocking=True) if 'pose' in data and use_pose_in_model else None
            target = target.to(device, non_blocking=True)

            if use_pose_in_model:
                output = model(clip, pose, tau_value)
            else:
                output = model(clip) # Adjust as per model's signature
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            prob = F.softmax(input=output, dim=1)

            batch_size = clip.shape[0]
            target_cpu = target.cpu().numpy() # Use a different variable name
            video_idx_cpu = video_idx.cpu().numpy() # Use a different variable name
            prob_cpu = prob.cpu().numpy() # Use a different variable name
            for i in range(batch_size):
                idx = video_idx_cpu[i]
                if idx in video_prob:
                    video_prob[idx] += prob_cpu[i]
                else:
                    video_prob[idx] = prob_cpu[i]
                    video_label[idx] = target_cpu[i]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    metric_logger.synchronize_between_processes()

    print(f' * Clip Acc@1 {metric_logger.acc1.global_avg:.3f} Clip Acc@5 {metric_logger.acc5.global_avg:.3f}')

    video_pred = {k: np.argmax(v) for k, v in video_prob.items()}
    pred_correct = [video_pred[k] == video_label[k] for k in video_pred if k in video_label] # Check if k exists

    if not pred_correct: # Handle empty pred_correct
        total_acc = 0.0
        print("Warning: No predictions made or labels available for video accuracy calculation.")
    else:
        total_acc = np.mean(pred_correct)

    num_classes = data_loader.dataset.num_classes if hasattr(data_loader.dataset, 'num_classes') and data_loader.dataset.num_classes > 0 else 0
    if num_classes > 0 :
        class_count = [0] * num_classes
        class_correct = [0] * num_classes

        for k, v_pred in video_pred.items(): # renamed v to v_pred for clarity
            if k in video_label:
                label = video_label[k]
                if 0 <= label < num_classes: # Check label bounds
                    class_count[label] += 1
                    class_correct[label] += (v_pred == label)
                else:
                    print(f"Warning: Label {label} out of bounds for num_classes {num_classes}")

        class_acc = [c / float(s) if s > 0 else 0 for c, s in zip(class_correct, class_count)]
        print(f' * Video Acc@1 {total_acc:.6f}')
        print(f' * Class Acc@1 {class_acc}')
    else:
        print(f' * Video Acc@1 {total_acc:.6f}')
        print("Warning: num_classes is 0 or not available, skipping Class Acc calculation.")


    return total_acc

def main(cfg, args):
    validate_data_paths(cfg)
    use_pose_in_model = cfg.data.add_points_around_pose.enabled

    # --- Dynamic Model Loading from Config ---
    model_class_name_from_cfg = cfg.model.name
    model_module_filename_from_cfg = cfg.model.module_filename # e.g., "sequence_mamba"

    module_path = f"models.{model_module_filename_from_cfg}" # Construct path like "models.sequence_mamba"

    # try:
    print(f"Attempting to load model class '{model_class_name_from_cfg}' from module '{module_path}'")
    ModelsModule = importlib.import_module(module_path)
    ModelClass = getattr(ModelsModule, model_class_name_from_cfg)
    print(f"Successfully loaded model: {model_class_name_from_cfg}")
    # except ImportError:
    #     print(f"Error: Could not import module '{module_path}'. Check the path and ensure the module exists.")
    #     sys.exit(1)
    # except AttributeError:
    #     print(f"Error: Class '{model_class_name_from_cfg}' not found in module '{module_path}'. Check the class name in your config and model file.")
    #     sys.exit(1)
    # except Exception as e:
    #     print(f"An unexpected error occurred during model loading: {e}")
    #     sys.exit(1)
    # --- End Dynamic Model Loading ---

    if cfg.training.output_dir:
        utils.mkdir(cfg.training.output_dir)
        log_file = os.path.join(cfg.training.output_dir, f'training_{time.strftime("%Y%m%d-%H%M%S")}.log')
        sys.stdout = Logger(log_file)

        # Save config and current script (train.py)
        shutil.copy(args.config, os.path.join(cfg.training.output_dir, os.path.basename(args.config)))
        shutil.copy(__file__, os.path.join(cfg.training.output_dir, os.path.basename(__file__))) # Save this script

    print(f"Loaded configuration: {vars(cfg)}") # Print config details
    print("PyTorch version:", torch.__version__)

    np.random.seed(cfg.data.seed)
    torch.manual_seed(cfg.data.seed)
    torch.cuda.manual_seed_all(cfg.data.seed) # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Dynamic Dataset Loading ---
    dataset_module_name = cfg.data.dataset_module # e.g., 'momo'
    dataset_class_name = cfg.data.dataset_name   # e.g., 'MomoDataset'
    try:
        DatasetModule = importlib.import_module(f"datasets.{dataset_module_name}")
        DatasetClass = getattr(DatasetModule, dataset_class_name)
    except (ImportError, AttributeError) as e:
        print(f"Error loading dataset: {dataset_class_name} from datasets.{dataset_module_name}. Exception: {e}")
        sys.exit(1)
    # --- End Dynamic Dataset Loading ---

    print("Loading data")
    dataset = DatasetClass(data_config=cfg.data, train=True)
    dataset_test = DatasetClass(data_config=cfg.data, train=False)

    print(f"Train dataset size: {len(dataset)}, Test dataset size: {len(dataset_test)}")
    if len(dataset) == 0 or len(dataset_test) == 0:
        print("Warning: One or both datasets are empty. Check data paths and dataset splitting logic.")

    print("Creating data loaders")
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.workers,
        pin_memory=True,
        drop_last=True
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.workers,
        pin_memory=True,
        drop_last=False
    )

    print("Creating model")
    # ModelClass is already loaded dynamically
    num_classes_for_model = dataset.num_classes if hasattr(dataset, 'num_classes') and dataset.num_classes > 0 else 30 # Fallback
    if num_classes_for_model == 0:
        print("Error: num_classes is 0. Cannot initialize model. Check dataset.")
        sys.exit(1)

    model = ModelClass(cfg, num_classes=num_classes_for_model) # Pass the whole cfg

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    if cfg.training.optimizer.type == 'AdamW':
        def add_weight_decay(model_to_decay, weight_decay=1e-5, skip_list=()):
            decay = []
            no_decay = []
            params_iter = model_to_decay.module.named_parameters() if isinstance(model_to_decay, nn.DataParallel) else model_to_decay.named_parameters()
            for name, param in params_iter:
                if not param.requires_grad:
                    continue
                if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay}]
        param_groups = add_weight_decay(model, weight_decay=cfg.training.optimizer.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=cfg.training.learning_rate.initial)
    elif cfg.training.optimizer.type == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.training.learning_rate.initial,
            weight_decay=cfg.training.optimizer.weight_decay
        )
    elif cfg.training.optimizer.type == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.training.learning_rate.initial,
            momentum=cfg.training.optimizer.momentum,
            weight_decay=cfg.training.optimizer.weight_decay,
            nesterov=True # Assuming nesterov=True is desired for SGD
        )
    else:
        raise NotImplementedError(f"Unsupported optimizer type: {cfg.training.optimizer.type}")

    # LR Scheduler
    t_warmup = cfg.training.learning_rate.warmup_epochs
    T_total = cfg.training.epochs
    n_t_cosine_scale = 0.5 # scale factor for cosine part
    min_lr_scale_factor = 0.01 # Minimum learning rate as a fraction of initial LR

    lambda1 = lambda epoch: (
        (1.0 - 0.1) * epoch / t_warmup + 0.1 # Linear warmup from 0.1 to 1.0
    ) if epoch < t_warmup else (
        min_lr_scale_factor + \
        (1.0 - min_lr_scale_factor) * n_t_cosine_scale * (1 + math.cos(math.pi * (epoch - t_warmup) / (T_total - t_warmup)))
    ) if T_total > t_warmup else 1.0 # if no cosine phase, keep lr at 1.0 * initial_lr

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)


    model_without_ddp = model.module if isinstance(model, nn.DataParallel) else model

    if cfg.training.resume and os.path.exists(cfg.training.resume): # Check if resume path exists
        print(f"Resuming from checkpoint: {cfg.training.resume}")
        checkpoint = torch.load(cfg.training.resume, map_location='cpu')

        state_dict = checkpoint['model']
        if all(key.startswith('module.') for key in state_dict.keys()) and not isinstance(model_without_ddp, nn.DataParallel):
             state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        elif not all(key.startswith('module.') for key in state_dict.keys()) and isinstance(model, nn.DataParallel):
             state_dict = {'module.' + k: v for k, v in state_dict.items()}

        model_without_ddp.load_state_dict(state_dict, strict=False) # Use strict=False if some layers are expected to change

        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if 'epoch' in checkpoint:
            cfg.training.start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming training from epoch {cfg.training.start_epoch}")
    else:
        if cfg.training.resume: # if path was given but not found
             print(f"Warning: Resume checkpoint not found at {cfg.training.resume}. Starting from scratch.")


    print("Start training")
    start_time = time.time()
    best_acc = 0.0 # Initialize best_acc
    current_tau = 1.0 # Initial tau if you're still using it this way

    for epoch in range(cfg.training.start_epoch, cfg.training.epochs):
        if cfg.training.epochs > 0 : # Avoid division by zero
             current_tau = 0.01 + (1.0 - 0.01) * 0.5 * (1 + np.cos(np.pi * epoch / cfg.training.epochs))
        else:
             current_tau = 1.0 # Default if epochs is 0

        print(f"Epoch {epoch}, Tau: {current_tau:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        train_one_epoch(
            model,
            criterion,
            optimizer,
            lr_scheduler,
            data_loader,
            device,
            epoch,
            cfg.training.print_freq,
            current_tau,
            use_pose_in_model # Pass this flag
        )

        current_eval_acc = evaluate(model, criterion, data_loader_test, device=device, tau_value=current_tau, use_pose_in_model=use_pose_in_model)

        if cfg.training.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'config_args': vars(args), # Save command line args (like --config path)
                'config_full': cfg # Save full config object (might be large if it contains complex objects)
            }
            utils.save_on_master(
                checkpoint,
                os.path.join(cfg.training.output_dir, 'checkpoint_last.pth')
            )

            if current_eval_acc > best_acc:
                print(f"New best accuracy: {current_eval_acc:.4f} (previous: {best_acc:.4f})")
                best_acc = current_eval_acc
                utils.save_on_master(
                    checkpoint,
                    os.path.join(cfg.training.output_dir, 'checkpoint_best.pth')
                )
        else: # If no output_dir, still track best_acc in memory
             if current_eval_acc > best_acc:
                best_acc = current_eval_acc


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')
    print(f'Best validation Accuracy {best_acc:.4f}')
    # Final evaluation with best model (optional, if you load it)
    if cfg.training.output_dir and os.path.exists(os.path.join(cfg.training.output_dir, 'checkpoint_best.pth')):
        print("Loading best model for final evaluation...")
        checkpoint_best = torch.load(os.path.join(cfg.training.output_dir, 'checkpoint_best.pth'), map_location=device)
        model_without_ddp.load_state_dict(checkpoint_best['model'])
        final_acc = evaluate(model, criterion, data_loader_test, device=device, tau_value=1.0, use_pose_in_model=use_pose_in_model) # Use a default tau for final test
        print(f'Final accuracy with best model: {final_acc:.4f}')


if __name__ == "__main__":
    args = parse_args_and_set_gpu()
    config = load_config(args.config)
    main(config, args)
