import torch
from txt_vis import TextVisualizer
from image_processor import TxtSpottingImageProcessor
import cv2
from tqdm import tqdm
# Common configurations
USE_TRT = True  # Set to True to use TensorRT, False to use PyTorch
IMAGE_PATH = "entry_mar_14_2024_01_00_53_pm_mar_14_2024_01_01_09_pm_fps_5_frame_34.png"
DIMS = (2048,1152)
INFERENCE_TH_TEST = 0.38
TRT_MODEL_PATH = '/home/ubuntu/models_deliver/txtspotting_r50_v0.1.1_2K.engine'

import asyncio


def main():
    processor = TxtSpottingImageProcessor(TRT_MODEL_PATH, DIMS)
    processed_results, original_image = processor.process_and_infer(IMAGE_PATH)
    instances = processed_results[0]["instances"].to(torch.device("cpu"))
    visualizer = TextVisualizer(original_image)
    vis_frame = visualizer.draw_instance_predictions(predictions=instances)
    cv2.imwrite('output_image.png', vis_frame)    

    # Benchmark speed
    num_iter = 100
    warm_up = 20
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iter)]    
    for i in tqdm(range(num_iter + warm_up)):
        if i >= warm_up:
            processor.benchmark_inference(IMAGE_PATH, benchmark_mode=True,
                                            iteration=i-warm_up, start_events=start_events, end_events=end_events)
        else:
            processor.benchmark_inference(IMAGE_PATH, benchmark_mode=False,
                                            iteration=i-warm_up, start_events=start_events, end_events=end_events)

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]    
    avg_t = sum(times)/num_iter
    print(f'Average latency: {str(avg_t)} ms')    




    
if __name__ == "__main__":
    main()
