"""Convert a 3-channel (RGB) detector checkpoint to 16-band input.

Finds the first 4-D weight with 3 input channels (the stem conv), averages it over colour and
tiles it across the 16 bands, scaled by 3/16 so activations keep their pretrained magnitude —
the same rule used for YOLO/RT-DETR (src/train.py:init_first_conv) and the D-FINE stems.

  python src/convert_ckpt_16ch.py weights/retinanet_r50_fpn_2x_coco.pth weights/retinanet_r50_16ch.pth
"""
import sys
import torch

def main(src, dst, bands=16):
    ck = torch.load(src, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    key = next(k for k, v in sd.items() if torch.is_tensor(v) and v.ndim == 4 and v.shape[1] == 3)
    w = sd[key]
    sd[key] = w.mean(1, keepdim=True).repeat(1, bands, 1, 1) * (3.0 / bands)
    print(f"{key}: {tuple(w.shape)} -> {tuple(sd[key].shape)}")
    if "state_dict" in ck:
        ck["state_dict"] = sd
        ck.pop("optimizer", None)      # stale optimiser state refers to the old shape
    torch.save(ck, dst)
    print("wrote", dst)

if __name__ == "__main__":
    main(*sys.argv[1:3])
