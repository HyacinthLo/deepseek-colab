"""Check that every linked shard has its published byte size. usage: python3 -I checksizes.py MODEL_DIR"""
import os, sys

SPECS = {
    'ista': ('DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-%05d-of-00006.gguf',
             {1: 5780704, 2: 39535781184, 3: 39736712000, 4: 39036908384, 5: 19923784832, 6: 110595290816}),
    'unsloth': ('DeepSeek-V4.1-Flash-1017-UD-IQ1_M-%05d-of-00005.gguf',
                {1: 5780868, 2: 49707332128, 3: 49292777280, 4: 40803165856, 5: 110595290592}),
}
weights = os.environ.get('WEIGHTS', 'ista')
if weights not in SPECS:
    sys.exit('unknown WEIGHTS %r (use ista or unsloth)' % weights)
pat, sizes = SPECS[weights]
bad = 0
for i, n in sizes.items():
    p = os.path.join(sys.argv[1], pat % i)
    s = os.path.getsize(p) if os.path.exists(p) else -1
    print(i, os.path.realpath(p), s, 'ok' if s == n else 'BAD')
    bad += s != n
sys.exit(1 if bad else 0)
