"""Parameter count and FLOPs for the submitted model (required by the competition rules).

  python code_package/report_flops.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "DEIM"))
import torch
from engine.core import YAMLConfig

CFG = "configs/hsi/hsi_deim_dfine_s_ms1536_all_e40.yml"
SHAPE = (1, 16, 768, 1536)

def main():
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "DEIM"))
    m = YAMLConfig(CFG).model
    n = sum(p.numel() for p in m.parameters())
    nt = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"parameters total     : {n:,} ({n/1e6:.2f} M)")
    print(f"parameters trainable : {nt:,} ({nt/1e6:.2f} M)")
    from calflops import calculate_flops
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    flops, macs, _ = calculate_flops(model=m.eval().to(dev), input_shape=SHAPE,
                                     output_as_string=True, output_precision=4, print_results=False)
    print(f"input shape          : {' x '.join(map(str, SHAPE))}")
    print(f"FLOPs                : {flops}")
    print(f"MACs                 : {macs}")

if __name__ == "__main__":
    main()
