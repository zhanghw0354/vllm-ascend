#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from typing import Optional, Tuple, Union, Any

import torch
import torch_npu
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.distributed import get_tp_group
from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
from vllm_ascend import envs


class AddRMSNormW8A8Quant(RMSNorm):
    # Fuse AddRmsNorm and W8A8 quantization ops together

    def __init__(
        self,
        hidden_size: int,
        layer: torch.nn.Module,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
        has_weight: bool = True,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(hidden_size, eps, var_hidden_size, has_weight, dtype)
        self.layer = layer

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            x, _, residual = torch_npu.npu_add_rms_norm_quant(
                x,
                residual,
                self.weight,
                self.layer.aclnn_input_scale,
                self.layer.aclnn_input_offset,
                epsilon=self.variance_epsilon)
            return x, residual

        x, residual = torch_npu.npu_rms_norm(x, self.weight,
                                             self.variance_epsilon)
        return x

class RMSNormFlashCommV1(RMSNorm):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
        module_name: Optional[str] = "",
    ) -> None:
        assert envs.VLLM_ASCEND_ENABLE_FLASHCOMM == 1, \
            f"VLLM_ASCEND_ENABLE_FLASHCOMM must be 1, but got {envs.VLLM_ASCEND_ENABLE_FLASHCOMM}."
        super().__init__(hidden_size, eps, var_hidden_size)
        self.module_name = module_name
        self.tp_size = get_tensor_model_parallel_world_size() # get tp size for each module
        self.tp_rank = get_tensor_model_parallel_rank() # get tp rank for each module

    def forward(
            self,
            x: torch.Tensor,
            residual: Optional[torch.Tensor] = None,
    ) -> Union[tuple[dict[str, Any], Any], Any]:
        if residual is not None:
            x, _, residual = torch_npu.npu_add_rms_norm(x, residual, self.weight, self.variance_epsilon)
            x = get_tp_group().all_gather(x, dim=0)
            return x, residual
        else:
            return torch_npu.npu_rms_norm(
                x,
                self.weight.data,
                self.variance_epsilon,
            )[0]

    def forward_with_residual(
            self,
            x: torch.Tensor,
            residual: Optional[torch.Tensor] = None,
            y_transform: str = "",
    ) -> Union[tuple[dict[str, Any], Any], Any]:
        if residual is not None:
            x, _, residual = torch_npu.npu_add_rms_norm(x, residual, self.weight, self.variance_epsilon)
            if y_transform == "AG":
                x = get_tp_group().all_gather(x, dim=0)
            return x, residual
        else:
            residual = x
            x = torch_npu.npu_rms_norm(
                x,
                self.weight.data,
                self.variance_epsilon,
            )[0]
            return x, residual

def forward_oot(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if residual is not None:
        x, _, residual = torch_npu.npu_add_rms_norm(x, residual, self.weight,
                                                    self.variance_epsilon)
        return x, residual

    x, residual = torch_npu.npu_rms_norm(x, self.weight, self.variance_epsilon)
    return x


RMSNorm.forward_oot = forward_oot
