# Vortex

Utilities for efficient inference and training of deep signal processing models.


## Inference

Set up the environment, then run:

```bash
python3 generate.py \
    --config_path /home/zymrael/workspace/stripedhyena-2/configs/shc-evo2-7b-8k-2T-v1.yml \
    --checkpoint_path /home/zymrael/checkpoints/evo2/7b_13h_8m_8s_3a_cascade15_inference/iter_457500.pt \
    --input_file prompt.txt \
    --cached_generation
```

The flag `--cached_generation` is optional, but recommended for faster generation. 

## Kernels



