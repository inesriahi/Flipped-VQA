# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import os
import time
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Callable
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from torch import inf
import numpy as np

@dataclass
class SmoothedValue:
    """
    Track and provide access to smoothed values over a window or the global series average.

    Attributes:
        window_size (int): Size of the window for calculating smoothed values. Default is 20.
        fmt (str, optional): Format string for output. Defaults to "{median:.4f} ({global_avg:.4f})".

    Example:
        smoothed_value = SmoothedValue(window_size=30)
        for value in data:
            smoothed_value.update(value)
        print(smoothed_value)
    """
    window_size: int = 20
    fmt: Optional[str] = "{median:.4f} ({global_avg:.4f})"
    _deque: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    total: float = 0.0
    count: int = 0

    def __post_init__(self):
        self._deque = deque(maxlen=self.window_size) # The maxlen attribute ensures that the deque only holds a maximum of window_size elements
        if self.fmt is None:
            self.fmt = "{median:.4f} ({global_avg:.4f})"

    def update(self, value: float, n: int =1) -> None:
        self._deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        This method ensures that the count and total attributes of the SmoothedValue instances are synchronized across all processes in distributed learning environment
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized(): # This line checks if the distributed environment is available and initialized. If it's not, the method simply returns without doing anything
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda') # The tensor is created on the CUDA device (GPU), assuming the environment supports CUDA. This is important for performance in distributed training scenarios. 
        dist.barrier() #  This is a synchronization point. It ensures that all processes reach this point before any of them proceeds. This is important to ensure that all processes perform the following all_reduce operation together.
        dist.all_reduce(t) # This is the key step for synchronization. The all_reduce operation aggregates the tensors across all processes by applying a specified operation (by default, the sum). After this operation, each process will have the sum of count and total from all processes.
        t = t.tolist()
        self.count = int(t[0]) # The local count and total attributes are updated based on the aggregated values. The count is converted to an integer.
        self.total = t[1]

    @property
    def median(self) -> float:
        d = torch.tensor(list(self._deque))
        return d.median().item()

    @property
    def avg(self) -> float: # The avg represents the average of a sliding window of the most recent values.
        d = torch.tensor(list(self._deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self) -> float: # this average is computed using the total sum of all the values
        if self.count == 0:
            # Handle the case where count is 0
            # Option 1: Return a default value, e.g., 0 or None
            return 0  # or None, or any other appropriate value

            # Option 2: Raise an error or warning
            # raise ValueError("Division by zero encountered in global_avg calculation")
    
        return self.total / self.count

    @property
    def max(self) -> float:
        return max(self._deque)

    @property
    def value(self) -> float:
        return self._deque[-1]

    def __str__(self) -> str:
        return self.fmt.format(median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value)


class MetricLogger:
    def __init__(self, delimiter: str = "\t") -> None:
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, count = 1, **metrics):
        for metric_name, value in metrics.items():
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                value = value.item()
            assert isinstance(value, (float, int))
            self.meters[metric_name].update(value, n=count)

    def __getattr__(self, name: str) -> Any:
        if name in self.meters:
            return self.meters[name]
        return super().__getattr__(name)

    def __str__(self) -> str:
        metric_str = [f"{name}: {meter}" for name, meter in self.meters.items()]
        return self.delimiter.join(metric_str)

    def synchronize_between_processes(self) -> None:
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name: str, meter: SmoothedValue) -> None:
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        index = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        num_digits = len(str(len(iterable)))
        space_fmt = f'{num_digits}d'
        log_msg = self._construct_log_msg(space_fmt, header, len(iterable))
        MB = 1024.0 * 1024.0
        for item in iterable:
            data_time.update(time.time() - end)
            yield item
            iter_time.update(time.time() - end)
            if index % print_freq == 0 or index == len(iterable) - 1:
                self._print_log(index, iterable, iter_time, data_time, log_msg, MB)
            index += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f'{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)')

    def _construct_log_msg(self, space_fmt: str, header: str, total: int) -> str:
        log_msg = [header, f'[{{:{space_fmt}}}/{total}]', 'eta: {eta}', '{meters}', 'time: {time}', 'data: {data}']
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        return self.delimiter.join(log_msg)

    def _print_log(self, index: int, iterable: Iterable, iter_time: SmoothedValue, data_time: SmoothedValue, log_msg: str, MB: float) -> None:
        eta_seconds = iter_time.global_avg * (len(iterable) - index)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        if torch.cuda.is_available():
            print(log_msg.format(index, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time), memory=torch.cuda.max_memory_allocated() / MB))
        else:
            print(log_msg.format(index, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)))

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    # torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, name):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / (f'{name}.pth')]
        
        unfrozen_model = {}
        for n, p in model_without_ddp.named_parameters():
            if ('gate' in n) or ('adapter' in n) or ('temporal_emb' in n) or ('visual_proj' in n):
                unfrozen_model[n] = p

        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': unfrozen_model,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


def all_reduce_mean(value: float) -> float:
    """
    Computes the average of a value across all nodes in a distributed system.

    This function takes a numeric value, synchronizes it across all nodes in the
    distributed system using a reduction operation, and then computes the average.

    Args:
        value (float): The value to be averaged across nodes.

    Returns:
        float: The average of the input value across all nodes.
    """
    world_size = get_world_size()
    if world_size <= 1:
        return value
    reduced_value = torch.tensor(value).cuda()
    dist.all_reduce(reduced_value)
    average_value = reduced_value / world_size
    return average_value.item()


def getCount(freq):
    count, total = freq[0], freq[1]
    return count / total if total != 0 else 0.0

def get_qtype_mapping(dataset_name: str) -> Dict[str, int]:
    if dataset_name == 'nextqa':
        return {'CH': 1, 'CW': 2, 'TN': 3, 'TC': 4, 'TP': 5, 'DL': 6, 'DC': 7, 'DO': 8}
    elif dataset_name == "star":
        return {'In': 1, 'Seq': 2, 'Pre': 3, 'Feas': 4}
    elif dataset_name == "valor32k":
        return {
            'count_visual': 1,
            'count_audio': 2,
            'count_both': 3,
            'temporal_visual': 4,
            'temporal_audio': 5,
            'temporal_both': 6,
            'desc_visual': 7,
            'desc_audio': 8,
            'desc_both': 9,
            'action_visual': 10,
            'action_audio': 11,
            'action_both': 12,
            'loc_visual': 13,
            'loc_audio': 14,
            'loc_both': 15,
            'rel_pos_visual': 16,
            'rel_pos_audio': 17,
            'rel_pos_both': 18,
            'audio_both': 19,
            'audio_visual': 20
            }
    elif dataset_name == "musicavqa":
        return {
            'Audio_Temporal': 1,
            'Audio_Existential': 2,
            'Audio_Comparative': 3,
            'Audio_Location': 4,
            'Audio_Counting': 5,
            'Visual_Temporal': 6,
            'Visual_Existential': 7,
            'Visual_Comparative': 8,
            'Visual_Location': 9,
            'Visual_Counting': 10,
            'Audio-Visual_Temporal': 11,
            'Audio-Visual_Existential': 12,
            'Audio-Visual_Comparative': 13,
            'Audio-Visual_Location': 14,
            'Audio-Visual_Counting': 15
        }
    else:
        return {}

def calculate_question_frequency(data, eval, qtype2id: Dict[str, int]) -> Dict[int, List[float]]:
    q_freq = {id: [0.0, 0.0] for id in qtype2id.values()}
    q_freq[0] = [0.0, 0.0]
    for i, v in enumerate(eval):
        qt = data['qtype'][i].item()
        q_freq[qt][0] += v.item()
        q_freq[qt][1] += 1
        q_freq[0][0] += v.item()
        q_freq[0][1] += 1
    return q_freq

def update_metrics_based_on_dataset(q_freq, metric_logger: MetricLogger, dataset_name: str, epsilon: float) -> None:
    """
    Updates the metrics based on the specific dataset.

    Args:
        q_freq: Dictionary containing the frequencies of different question types.
        metric_logger: Logger for recording metrics.
        dataset_name: Name of the dataset (e.g., 'nextqa', 'star').
        epsilon: A small number to avoid division by zero.
    """
    if dataset_name == 'nextqa':
        update_nextqa_metrics(q_freq, metric_logger, epsilon)
    elif dataset_name == "star":
        update_star_metrics(q_freq, metric_logger, epsilon)
    elif dataset_name == "valor32k":
        update_valor32k_metrics(q_freq, metric_logger, epsilon)
    elif dataset_name == "musicavqa":
        update_musicavqa_metrics(q_freq, metric_logger, epsilon)

def update_nextqa_metrics(q_freq, metric_logger, epsilon: float) -> None:
    # Logic specific to the 'nextqa' dataset
    metric_logger.update(n=(q_freq[1][1] + q_freq[2][1] + epsilon), C=(q_freq[1][0] + q_freq[2][0]) / (q_freq[1][1] + q_freq[2][1] + epsilon))
    metric_logger.update(n=(q_freq[3][1] + q_freq[4][1] + q_freq[5][1] + epsilon), T=(q_freq[3][0] + q_freq[4][0] + q_freq[5][0]) / (q_freq[3][1] + q_freq[4][1] + q_freq[5][1] + epsilon))
    metric_logger.update(n=(q_freq[6][1] + q_freq[7][1] + q_freq[8][1] + epsilon), D=(q_freq[6][0] + q_freq[7][0] + q_freq[8][0]) / (q_freq[6][1] + q_freq[7][1] + q_freq[8][1] + epsilon))
    metric_logger.update(n=q_freq[0][1] + epsilon, Total=getCount(q_freq[0]))


def update_star_metrics(q_freq, metric_logger: MetricLogger, epsilon: float) -> None:
    # Logic specific to the 'star' dataset
    metric_logger.update(n=q_freq[1][1] + epsilon, In=getCount(q_freq[1]))
    metric_logger.update(n=q_freq[2][1] + epsilon, Seq=getCount(q_freq[2]))
    metric_logger.update(n=q_freq[3][1] + epsilon, Pre=getCount(q_freq[3]))
    metric_logger.update(n=q_freq[4][1] + epsilon, Feas=getCount(q_freq[4]))
    metric_logger.update(n=q_freq[0][1] + epsilon, Total=getCount(q_freq[0]))

def update_valor32k_metrics(q_freq, metric_logger: MetricLogger, epsilon: float) -> None:
    def calculate_score_and_count(ids):
        total_score = sum([q_freq[id][0] for id in ids])
        total_count = sum([q_freq[id][1] for id in ids])
        avg_score = total_score / (total_count + epsilon) #if total_count > 0 else None
        return avg_score, total_count

    # Categories
    audio_score, audio_count = calculate_score_and_count([2, 5, 8, 11, 14, 17])
    visual_score, visual_count = calculate_score_and_count([1, 4, 7, 10, 13, 16, 20])
    both_score, both_count = calculate_score_and_count([3, 6, 9, 12, 15, 18, 19])

    # Question Types
    count_score, count_count = calculate_score_and_count([1, 2, 3])
    temporal_score, temporal_count = calculate_score_and_count([4, 5, 6])
    desc_score, desc_count = calculate_score_and_count([7, 8, 9])
    action_score, action_count = calculate_score_and_count([10, 11, 12])
    loc_score, loc_count = calculate_score_and_count([13, 14, 15])
    rel_pos_score, rel_pos_count = calculate_score_and_count([16, 17, 18])
    audio_second_score, audio_second_count = calculate_score_and_count([19, 20])

    # Update metrics
    metric_logger.update(
        n_audio=audio_count + epsilon, audio=audio_score, 
        n_visual=visual_count + epsilon, visual=visual_score, 
        n_both=both_count + epsilon, both=both_score,
        n_count=count_count + epsilon, count=count_score, 
        n_temporal=temporal_count + epsilon, temporal=temporal_score, 
        n_desc=desc_count + epsilon, desc=desc_score, 
        n_action=action_count + epsilon, action=action_score, 
        n_loc=loc_count + epsilon, loc=loc_score, 
        n_rel_pos=rel_pos_count + epsilon, rel_pos=rel_pos_score,
        n_audio_second=audio_second_count + epsilon, audio_second=audio_second_score
    )

def update_musicavqa_metrics(q_freq, metric_logger: MetricLogger, epsilon: float) -> None:
    def calculate_score_and_count(ids):
        total_score = sum([q_freq[id][0] for id in ids])
        total_count = sum([q_freq[id][1] for id in ids])
        avg_score = total_score / (total_count + epsilon)
        return avg_score, total_count

    # Categories based on sensory modality
    audio_score, audio_count = calculate_score_and_count([1, 2, 3, 4, 5])
    visual_score, visual_count = calculate_score_and_count([6, 7, 8, 9, 10])
    audio_visual_score, audio_visual_count = calculate_score_and_count([11, 12, 13, 14, 15])

    # Categories based on question type
    temporal_score, temporal_count = calculate_score_and_count([1, 6, 11])
    existential_score, existential_count = calculate_score_and_count([2, 7, 12])
    comparative_score, comparative_count = calculate_score_and_count([3, 8, 13])
    location_score, location_count = calculate_score_and_count([4, 9, 14])
    counting_score, counting_count = calculate_score_and_count([5, 10, 15])

    # Update metrics
    metric_logger.update(
        n_audio=audio_count + epsilon, audio=audio_score,
        n_visual=visual_count + epsilon, visual=visual_score,
        n_audio_visual=audio_visual_count + epsilon, audio_visual=audio_visual_score,
        n_temporal=temporal_count + epsilon, temporal=temporal_score,
        n_existential=existential_count + epsilon, existential=existential_score,
        n_comparative=comparative_count + epsilon, comparative=comparative_score,
        n_location=location_count + epsilon, location=location_score,
        n_counting=counting_count + epsilon, counting=counting_score
    )

def log_qtype(data, eval, metric_logger: MetricLogger, args):
    epsilon = 1e-10
    qtype2id = get_qtype_mapping(args.dataset)
    if not qtype2id:
        return
    question_frequency = calculate_question_frequency(data, eval, qtype2id)
    update_metrics_based_on_dataset(question_frequency, metric_logger, args.dataset, epsilon)

# def filter_result(result):
#     new_result = []
#     all_images = []
#     for d in result:
#         if d["image_id"] not in all_images:
#             all_images.append(d["image_id"])
#             new_result.append(d)
#         else:
#             continue
#     return new_result

def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    else:
        return obj

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)
        
def save_result(result, result_dir, filename, is_json=True, is_list=True):
    if is_json:
        result_file = os.path.join(result_dir, '%s_rank%d.json' % (filename, get_rank()))
        final_result_file = os.path.join(result_dir, '%s.json' % filename)
        with open(result_file, 'w') as f:
            json.dump(result, f, cls=NumpyEncoder)
    else:
        result_file = os.path.join(result_dir, '%s_rank%d.pth' % (filename, get_rank()))
        final_result_file = os.path.join(result_dir, '%s.pth' % filename)
        torch.save(result, result_file)

    dist.barrier()

    if is_main_process():
        # combine results from all processes
        if is_list:
            result = []
        else:
            result = {}
        for rank in range(get_world_size()):
            if is_json:
                result_file = os.path.join(result_dir, '%s_rank%d.json' % (filename, rank))
                res = json.load(open(result_file, 'r'))
            else:
                result_file = os.path.join(result_dir, '%s_rank%d.pth' % (filename, rank))
                res = torch.load(result_file)
            if is_list:
                result += res
            else:
                result.update(res)
        if is_json:
            # new_result = filter_result(result)
            # Use the custom encoder when serializing
            with open(result_file, 'w') as f:
                json.dump(result, f, cls=NumpyEncoder)
        else:
            torch.save(result, final_result_file)

        print('result file saved to %s' % final_result_file)
    dist.barrier()
    return final_result_file