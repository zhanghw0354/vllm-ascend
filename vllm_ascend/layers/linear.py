from typing import List, Optional
from abc import abstractmethod

import torch
import torch_npu
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.linear import LinearMethodBase

from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.utils import set_weight_attrs

from vllm.distributed import (
    divide,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
    get_tp_group
)

class FlashCommLinearMethodBase(LinearMethodBase):
    """Base class for different (maybe quantized) linear methods."""

    @abstractmethod
    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None,
              module_name: Optional[str] = "",
              x_transform: Optional[str] = None) -> torch.Tensor:
        """Apply the weights in layer to the input tensor.
        Expects create_weights to have been called before on the layer."""
        raise NotImplementedError

class UnquantizedFlashCommLinearMethod(FlashCommLinearMethodBase):
    """Linear method without quantization."""

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        weight = Parameter(torch.empty(sum(output_partition_sizes),
                                       input_size_per_partition,
                                       dtype=params_dtype),
                           requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        weight_data = layer.weight.data.t().contiguous()
        layer.weight.data = weight_data
        set_weight_attrs(layer.weight, {"is_weight_transposed": True})
        # weight_data = torch_npu.npu_format_cast(layer.weight.data.t().contiguous(), 29)
        # layer.weight = torch.nn.Parameter(weight_data, requires_grad=False)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None,
              module_name: Optional[str] = "",
              x_transform: Optional[str] = None,
              is_prefill: Optional[bool] = True) -> torch.Tensor:
        
        if x_transform == "AG":
            x = get_tp_group().all_gather(x, dim=0)
        elif x_transform == "A2A":
            x = get_tp_group().all_to_all(x)

        if bias is not None:
            # return F.linear(x, layer.weight, bias)
            return torch.addmm(bias, x, layer.weight)
        else:
            return torch.matmul(x, layer.weight)

class FlashCommLinearBase(torch.nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.skip_bias_add = skip_bias_add
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        if quant_config is None:
            self.quant_method: Optional[
                QuantizeMethodBase] = UnquantizedFlashCommLinearMethod()
        else:
            self.quant_method = quant_config.get_quant_method(self,
                                                              prefix=prefix)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

"""
Similar to vllm's original RowParallelLinear except for:
1. layerwise TP configurations. call get_tensor_model_parallel_world_size/rank.
2. flexible communication applied to x or y during forward (controled by parameters of forward function).
"""
class RowParallelFlashCommLinear(FlashCommLinearBase):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 tp_size: int = 1,
                 tp_rank: int = 0,
                 bias: bool = True,
                 skip_bias_add: bool = False,
                 params_dtype: Optional[torch.dtype] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__(input_size, output_size, tp_size, tp_rank, skip_bias_add, params_dtype,
                         quant_config, prefix)
        # Divide the weight matrix along the first dimension.
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.prefix = prefix

        assert self.quant_method is not None
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=[self.output_size],
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=self.weight_loader)

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size, dtype=params_dtype))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        # veRL special case: transpose the weight back to original shape
        is_weight_transposed = getattr(param, "is_weight_transposed", False)
        if is_weight_transposed:
            param.data = param.data.t().contiguous()
        input_dim = getattr(param, "input_dim", None)
        param_data = param.data

        if input_dim is not None:
            shard_size = param_data.shape[input_dim]
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(input_dim, start_idx,
                                                 shard_size)

        loaded_weight = torch.squeeze(loaded_weight)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)
        # veRL special case: transpose the weight to use torch npu operator
        if is_weight_transposed:
            param.data = param.data.t().contiguous()

    def forward(self, input_, reduce_type="AR", x_transform=None):
        input_parallel = input_

        # Matrix multiply.
        assert self.quant_method is not None
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method.apply(self,
                                                  input_parallel,
                                                  bias=bias_,
                                                  module_name=self.prefix,
                                                  x_transform=x_transform)
        if self.tp_size > 1:
            if reduce_type == "AR":
                output = tensor_model_parallel_all_reduce(output_parallel)
            elif reduce_type == "RS":
                output = tensor_model_parallel_reduce_scatter(output_parallel)
            else:
                output = output_parallel
        else:
            output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None

        return output, output_bias

class FlashCommLinearBase(torch.nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.skip_bias_add = skip_bias_add
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        if quant_config is None:
            self.quant_method: Optional[
                QuantizeMethodBase] = UnquantizedFlashCommLinearMethod()
        else:
            self.quant_method = quant_config.get_quant_method(self,
                                                              prefix=prefix)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class ColumnParallelFlashCommLinear(FlashCommLinearBase):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 tp_size: int = 1,
                 tp_rank: int = 0,
                 bias: bool = True,
                 skip_bias_add: bool = False,
                 params_dtype: Optional[torch.dtype] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 output_sizes: Optional[List[int]] = None,
                 prefix: str = ""):
        super().__init__(input_size, output_size, tp_size, tp_rank, skip_bias_add, params_dtype,
                         quant_config, prefix)

        # Divide the weight matrix along the last dimension.
        assert self.quant_method is not None
        self.output_size_per_partition = divide(self.output_size, self.tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        # If QKV or MergedColumn, use output size of each partition.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                divide(output_size, self.tp_size)
                for output_size in self.output_sizes
            ]

        if output_sizes is None:
            output_sizes = [output_size]

        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=self.weight_loader)
        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size_per_partition,
                            dtype=params_dtype))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)
        self.prefix = prefix

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        # veRL special case: transpose the weight back to original shape
        is_weight_transposed = getattr(param, "is_weight_transposed", False)
        if is_weight_transposed:
            param.data = param.data.t().contiguous()
        output_dim = getattr(param, "output_dim", None)

        param_data = param.data
        if output_dim is not None:
            shard_size = param_data.shape[output_dim]
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                 shard_size)

        loaded_weight = torch.squeeze(loaded_weight)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)
        # veRL special case: transpose the weight to use torch npu operator
        if is_weight_transposed:
            param.data = param.data.t().contiguous()

    def forward(self, input_, x_transform=None, is_prefill=True):
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        assert self.quant_method is not None
        output = self.quant_method.apply(self, input_, bias, module_name=self.prefix, x_transform=x_transform, is_prefill=is_prefill)
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def extra_repr(self) -> str:
        s = f"in_features={self.input_size}"
        s += f", output_features={self.output_size_per_partition}"
        s += f", bias={self.bias is not None}"
        s += f", tp_size={self.tp_size}"
        return s

class QKVParallelFlashCommLinear(ColumnParallelFlashCommLinear):

    def __init__(self,
                 hidden_size: int,
                 head_size: int,
                 total_num_heads: int,
                 total_num_kv_heads: Optional[int] = None,
                 tp_size: int = 1,
                 tp_rank: int = 0,
                 bias: bool = True,
                 skip_bias_add: bool = False,
                 params_dtype: Optional[torch.dtype] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        # Divide the weight matrix along the last dimension.
        self.prefix = prefix
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size,
                                               self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        input_size = self.hidden_size
        output_size = (self.num_heads +
                       2 * self.num_kv_heads) * tp_size * self.head_size
        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,  # q_proj
            self.num_kv_heads * self.head_size * tp_size,  # k_proj
            self.num_kv_heads * self.head_size * tp_size,  # v_proj
        ]

        super().__init__(input_size=input_size,
                         output_size=output_size,
                         tp_size=tp_size,
                         tp_rank=tp_rank,
                         bias=bias,
                         skip_bias_add=skip_bias_add,
                         params_dtype=params_dtype,
                         quant_config=quant_config,
                         prefix=prefix)

    def weight_loader(self,
                      param: Parameter,
                      loaded_weight: torch.Tensor,
                      loaded_shard_id: Optional[str] = None):
        # veRL special case: transpose the weight back to original shape
        is_weight_transposed = getattr(param, "is_weight_transposed", False)
        if is_weight_transposed:
            param.data = param.data.t().contiguous()
        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        assert loaded_shard_id in ["q", "k", "v"]

        # If output dim is defined, use the default loading process.
        assert output_dim is not None
        if loaded_shard_id == "q":
            shard_offset = 0
            shard_size = self.num_heads * self.head_size
        elif loaded_shard_id == "k":
            shard_offset = self.num_heads * self.head_size
            shard_size = self.num_kv_heads * self.head_size
        elif loaded_shard_id == "v":
            shard_offset = (self.num_heads +
                            self.num_kv_heads) * self.head_size
            shard_size = self.num_kv_heads * self.head_size

        param_data = param_data.narrow(output_dim, shard_offset,
                                        shard_size)
        if loaded_shard_id == "q":
            shard_id = self.tp_rank
        else:
            shard_id = self.tp_rank // self.num_kv_head_replicas
        start_idx = shard_id * shard_size

        loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                shard_size)

        loaded_weight = torch.squeeze(loaded_weight)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)
        # veRL special case: transpose the weight to use torch npu operator
        if is_weight_transposed:
            param.data = param.data.t().contiguous()