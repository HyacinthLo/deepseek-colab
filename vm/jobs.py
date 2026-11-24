"""Writes the pfetch job list of one download (backbone shards, Engram or DSpark draft) at pinned HF revisions. usage: python3 jobs.py backbone|engram|draft DEST_DIR OUT.json"""
import json, os, sys

SPECS = {
    'ista': {
        'hf': 'https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/resolve/a419d74cd28d5b9f8c5e2b9077d11a609c17bc32/',
        'name': 'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-%05d-of-00006.gguf',
        'sizes': [5780704, 39535781184, 39736712000, 39036908384, 19923784832],
        'engram': ('DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf', 110595290816),
    },
    'unsloth': {
        'hf': 'https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/resolve/dfcee0ec8d6b874c6663eca0fc55d90ab32534c8/UD-IQ1_M/',
        'name': 'DeepSeek-V4.1-Flash-1017-UD-IQ1_M-%05d-of-00005.gguf',
        'sizes': [5780868, 49707332128, 49292777280, 40803165856],
        'engram': ('DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00005-of-00005.gguf', 110595290592),
    },
}
DRAFT = 'https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/resolve/85f81fb970add013033f25e94006713ca2c0aaf2/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf'

weights = os.environ.get('WEIGHTS', 'ista')
if weights not in SPECS:
    sys.exit('unknown WEIGHTS %r (use ista or unsloth)' % weights)
sp = SPECS[weights]
what, dest, out = sys.argv[1:4]
files = {'backbone': [(sp['hf'] + sp['name'] % (i + 1), sp['name'] % (i + 1), s) for i, s in enumerate(sp['sizes'])],
         'engram': [(sp['hf'] + sp['engram'][0], sp['engram'][0], sp['engram'][1])],
         'draft': [(DRAFT, 'dspark.gguf', 7976914240)]}[what]
with open(out, 'w') as f:
    json.dump([{'url': u, 'dest': os.path.join(dest, n), 'dest_size': s, 'src_off': 0, 'dest_off': 0, 'size': s} for u, n, s in files], f)
print('%s (%s): %.2f GB' % (what, weights, sum(s for _, _, s in files) / 1e9))
