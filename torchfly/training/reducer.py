import torch
import torch.nn as nn
import torch.distributed as dist
from collections import OrderedDict

imported_flatten_impl = False


def import_flatten_impl():
    global flatten_impl, unflatten_impl, imported_flatten_impl
    try:
        import apex_C
        flatten_impl = apex_C.flatten
        unflatten_impl = apex_C.unflatten
    except ImportError:
        print("Warning:  apex was installed without --cpp_ext.  Falling back to Python flatten and unflatten.")
        flatten_impl = torch._utils._flatten_dense_tensors
        unflatten_impl = torch._utils._unflatten_dense_tensors
    imported_flatten_impl = True


def flatten(bucket):
    if not imported_flatten_impl:
        import_flatten_impl()
    return flatten_impl(bucket)


def unflatten(coalesced, bucket):
    if not imported_flatten_impl:
        import_flatten_impl()
    return unflatten_impl(coalesced, bucket)


def apply_flat_dist_call(bucket, call, extra_args=None):

    coalesced = flatten(bucket)

    if extra_args is not None:
        call(coalesced, *extra_args)
    else:
        call(coalesced)

    if call is dist.all_reduce:
        coalesced /= dist.get_world_size()

    for buf, synced in zip(bucket, unflatten(coalesced, bucket)):
        buf.copy_(synced)


def split_by_type(tensors):
    buckets = OrderedDict()
    for tensor in tensors:
        tp = tensor.type()
        if tp not in buckets:
            buckets[tp] = []
        buckets[tp].append(tensor)
    return buckets


# flat_dist_call organizes 'tensors' by type.
def flat_dist_call(tensors, call, extra_args=None):
    buckets = split_by_type(tensors)

    for tp in buckets:
        bucket = buckets[tp]
        apply_flat_dist_call(bucket, call, extra_args)


def extract_tensors(maybe_tensor, tensor_list):
    if torch.is_tensor(maybe_tensor):
        tensor_list.append(maybe_tensor)
    else:
        try:
            for item in maybe_tensor:
                extract_tensors(item, tensor_list)
        except TypeError:
            return


class Reducer:
    """
    :class:`apex.parallel.Reducer` is a simple class that helps allreduce a module's parameters
    across processes.  :class:`Reducer` is intended to give the user additional control:
    Unlike :class:`DistributedDataParallel`, :class:`Reducer` will not automatically allreduce
    parameters during ``backward()``.
    Instead, :class:`Reducer` waits for the user to call ``<reducer_instance>.reduce()`` manually.
    This enables, for example, delaying the allreduce to be carried out every
    several iterations instead of every single iteration.
    Like :class:`DistributedDataParallel`, :class:`Reducer` averages any tensors it allreduces
    over the number of participating processes.
    :class:`Reducer` is designed to work with the upstream launch utility script
    ``torch.distributed.launch`` with ``--nproc_per_node <= number of gpus per node``.
    When used with this launcher, :class:`Reducer` assumes 1:1 mapping of processes to GPUs.
    It also assumes that your script calls ``torch.cuda.set_device(args.rank)`` before creating the model.
    Args:
        module_or_grads_list: Either a network definition (module) being run in multi-gpu/distributed mode, or an iterable of gradients to be reduced.  If a module is passed in, the Reducer constructor will sync the parameters across processes (broadcasting from rank 0) to make sure they're all initialized with the same values.  If a list of gradients (that came from some module) is passed in, the user is responsible for manually syncing that module's parameters at the beginning of training.
    """

    def __init__(self, module_or_grads_list):
        if isinstance(module_or_grads_list, nn.Module):
            self.module = module_or_grads_list
            flat_dist_call([param.data for param in self.module.parameters()], dist.broadcast, (0,))
        else:
            self.module = None
            self.grads = []
            extract_tensors(module_or_grads_list, self.grads)

    def reduce(self):
        if self.module:
            grads = [param.grad.data for param in self.module.parameters() if param.grad is not None]
            flat_dist_call(grads, dist.all_reduce)
        else:
            flat_dist_call(self.grads, dist.all_reduce)
