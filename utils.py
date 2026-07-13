import datetime
import errno
import os
import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist

from modules import pointnet2_utils


def fps(data, number):
    """Sample ``number`` points from a batched point cloud."""
    indices = pointnet2_utils.furthest_point_sample(data, number)
    features = pointnet2_utils.gather_operation(
        data.transpose(1, 2).contiguous(),
        indices,
    )
    return features.transpose(1, 2).contiguous()


class SmoothedValue:
    """Track recent values and their global average."""

    def __init__(self, window_size=20, fmt=None):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt or "{median:.4f} ({global_avg:.4f})"

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        values = torch.tensor(
            [self.count, self.total],
            dtype=torch.float64,
            device="cuda",
        )
        dist.barrier()
        dist.all_reduce(values)
        self.count = int(values[0].item())
        self.total = values[1].item()

    @property
    def median(self):
        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self):
        return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for name, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            if not isinstance(value, (float, int)):
                raise TypeError(f"metric {name} must be numeric")
            self.meters[name].update(value)

    def __getattr__(self, attribute):
        if attribute in self.meters:
            return self.meters[attribute]
        raise AttributeError(
            f"{type(self).__name__!s} object has no attribute {attribute!r}"
        )

    def __str__(self):
        return self.delimiter.join(
            f"{name}: {meter}" for name, meter in self.meters.items()
        )

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        header = header or ""
        start_time = time.time()
        end_time = time.time()
        iteration_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        index_width = ":" + str(len(str(len(iterable)))) + "d"
        fields = [
            header,
            "[{0" + index_width + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            fields.append("max mem: {memory:.0f}")
        message = self.delimiter.join(fields)
        megabyte = 1024.0 * 1024.0

        for index, item in enumerate(iterable):
            data_time.update(time.time() - end_time)
            yield item
            iteration_time.update(time.time() - end_time)
            if index % print_freq == 0:
                eta_seconds = iteration_time.global_avg * (len(iterable) - index)
                values = {
                    "eta": str(datetime.timedelta(seconds=int(eta_seconds))),
                    "meters": str(self),
                    "time": str(iteration_time),
                    "data": str(data_time),
                }
                if torch.cuda.is_available():
                    values["memory"] = torch.cuda.max_memory_allocated() / megabyte
                print(message.format(index, len(iterable), **values))
            end_time = time.time()

        elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        print(f"{header} Total time: {elapsed}")


def accuracy(output, target, topk=(1,)):
    """Compute top-k accuracy percentages."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, predictions = output.topk(maxk, 1, True, True)
        correct = predictions.t().eq(target[None])
        return [
            correct[:k].flatten().sum(dtype=torch.float32) * (100.0 / batch_size)
            for k in topk
        ]


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as error:
        if error.errno != errno.EEXIST:
            raise


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


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
