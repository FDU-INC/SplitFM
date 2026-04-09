import os
import argparse

try:
    from .runtime import MODEL_REGISTRY, load_adapter
except ImportError:
    from runtime import MODEL_REGISTRY, load_adapter

def main():
    parser = argparse.ArgumentParser(description="SplitInfer Unified Interface")
    parser.add_argument("--model_name", type=str, required=True, choices=MODEL_REGISTRY.keys())
    parser.add_argument("--input_sentence", type=str, required=True)
    parser.add_argument("--image", type=str, default=None, help="Image path (for Qwen2-VL-7B-Instruct)")
    parser.add_argument("--gpu", type=str, default="0,1,2")
    args = parser.parse_args()

    # Set the visible CUDA devices (GPUs) for PyTorch
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # load 
    adapter, weights_path = load_adapter(args.model_name)
    adapter.load(weights_path)
    
    # inference
    result = adapter.infer(args.input_sentence, image_path=args.image)
    print(result)

if __name__ == "__main__":
    main()
