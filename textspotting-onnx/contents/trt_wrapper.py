"""
TRT wrapper using pytorch module.
Copied from [e2e-pipeline](https://github.com/terminal-industries/e2e-ml-pipeline/blob/main/src/modules/trt_wrapper.py)
polygraphy convert deepsolo_r50.onnx -o deepsolo_r50.engine --trt-config-script trt_config.py --trt-config-func-name set_trt_config
"""

from typing import Dict, Optional, Sequence, Union

import tensorrt as trt
import torch


def torch_dtype_from_trt(dtype: trt.DataType) -> torch.dtype:
    """Convert pytorch dtype to TensorRT dtype.

    Args:
        dtype (str.DataType): The data type in tensorrt.

    Returns:
        torch.dtype: The corresponding data type in torch.
    """

    if dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int8:
        return torch.int8
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError(f"{dtype} is not supported by torch")


class TRTWrapper(torch.nn.Module):
    def __init__(
        self,
        engine: Union[str, trt.ICudaEngine],
        output_names: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.engine = engine
        print(self.engine)
        if isinstance(self.engine, str):
            with trt.Logger() as logger, trt.Runtime(logger) as runtime:
                with open(self.engine, mode="rb") as f:
                    engine_bytes = f.read()                    
                self.engine = runtime.deserialize_cuda_engine(engine_bytes)
                print(self.engine)
                if self.engine is None:
                    raise RuntimeError("Failed to create execution context.")        

        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context        
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        
        #input_names = [name for name in names if self.engine.binding_is_input(self.engine.get_binding_index(name))]
        input_names = ['image']
        self._input_names = input_names
        self._output_names = None 
        
        #names = [_ for _ in self.engine]
        #input_names = list(filter(self.engine.binding_is_input, names))
        #self._input_names = input_names
        #self._output_names = output_names

        if self._output_names is None:
            output_names = list(set(names) - set(input_names))
            self._output_names = output_names
        
    '''
    def forward(self, inputs: Dict[str, torch.Tensor]):
        assert self._input_names is not None
        assert self._output_names is not None
        bindings = [None] * (len(self._input_names) + len(self._output_names))
        profile_id = 0

        for input_name, input_tensor in inputs.items():
            # check if input shape is valid
            idx = self.engine.get_binding_index(input_name)
            
            if idx == -1:
                raise ValueError(f"Input name {input_name} not found in engine bindings.")
            profile = self.engine.get_profile_shape(profile_id, idx)

            assert input_tensor.dim() == len(
                profile[0]
            ), "Input dim is different from engine profile."
            for s_min, s_input, s_max in zip(
                profile[0], input_tensor.shape, profile[2]
            ):
                assert s_min <= s_input <= s_max, (
                    "Input shape should be between "
                    + f"{profile[0]} and {profile[2]}"
                    + f" but get {tuple(input_tensor.shape)}."
                )
            idx = self.engine.get_binding_index(input_name)

            # All input tensors must be gpu variables
            assert "cuda" in input_tensor.device.type
            input_tensor = input_tensor.contiguous()
            if input_tensor.dtype == torch.long:
                input_tensor = input_tensor.int()
            self.context.set_binding_shape(idx, tuple(input_tensor.shape))
            bindings[idx] = input_tensor.contiguous().data_ptr()

        # create output tensors
        outputs = {}
        for output_name in self._output_names:
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            shape = tuple(self.context.get_binding_shape(idx))

            device = torch.device("cuda")
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[output_name] = output
            bindings[idx] = output.data_ptr()
        self.context.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)
        
        ctrl_point_cls = outputs['ctrl_point_cls'].detach().cpu().numpy()
        ctrl_point_coord = outputs['ctrl_point_coord'].detach().cpu().numpy()
        ctrl_point_text = outputs['ctrl_point_text'].detach().cpu().numpy()
        bd_points = outputs['bd_pointsoutput'].detach().cpu().numpy()
        predictions2 = [ctrl_point_cls,ctrl_point_coord,ctrl_point_text,bd_points]

        return predictions2
    '''
    def forward(self, inputs: Dict[str, torch.Tensor]):
        assert self._input_names is not None
        assert self._output_names is not None
        bindings = [None] * (len(self._input_names) + len(self._output_names))
        profile_id = 0

        for input_name, input_tensor in inputs.items():
            # check if input shape is valid
            #idx = self.engine.get_binding_index(input_name)
            
            #if idx == -1:
            #    raise ValueError(f"Input name {input_name} not found in engine bindings.")
            #profile = self.engine.get_profile_shape(profile_id, idx)
            profile = self.engine.get_tensor_profile_shape(input_name, profile_id)
            
            assert input_tensor.dim() == len(
                profile[0]
            ), "Input dim is different from engine profile."
            for s_min, s_input, s_max in zip(
                profile[0], input_tensor.shape, profile[2]
            ):
                assert s_min <= s_input <= s_max, (
                    "Input shape should be between "
                    + f"{profile[0]} and {profile[2]}"
                    + f" but get {tuple(input_tensor.shape)}."
                )
            idx = self.engine.get_binding_index(input_name)

            # All input tensors must be gpu variables
            assert "cuda" in input_tensor.device.type
            input_tensor = input_tensor.contiguous()
            if input_tensor.dtype == torch.long:
                input_tensor = input_tensor.int()
            self.context.set_binding_shape(idx, tuple(input_tensor.shape))
            bindings[idx] = input_tensor.contiguous().data_ptr()

        # create output tensors
        outputs = {}
        for output_name in self._output_names:
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            shape = tuple(self.context.get_binding_shape(idx))

            device = torch.device("cuda")
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[output_name] = output
            bindings[idx] = output.data_ptr()
        self.context.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)

        return outputs        