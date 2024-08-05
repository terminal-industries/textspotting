import tensorrt as trt

def set_trt_config(builder, network):
    config = builder.create_builder_config()
    
    # For TensorRT 8.2 or later
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16 * 1024 * 1024 * 1024)
    
    # Add any other necessary configuration settings here
    
    return config
