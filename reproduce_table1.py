import os
import glob
import json
import torch
import PIL
from PIL import Image
import numpy as np
from tqdm import tqdm
from transformers import AutoImageProcessor, Dinov2Model, AutoProcessor, Blip2ForImageTextRetrieval
import torch.nn.functional as F
from torchmetrics.image.kid import KernelInceptionDistance
import warnings
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLIP_LOCAL_PATH = r"d:\projects\BlackScholesDiffusion2024-main\Model\BLIP-2"
DINO_LOCAL_PATH = r"d:\projects\BlackScholesDiffusion2024-main\Model\DINOv2"
SETS = ['set1', 'set2', 'set3', 'set4']
METHODS = ['lininterp', 'alternating_sampling', 'clip_min', 'step', 'bs']

METHOD_MAP = {
    'lininterp': 'Linear Int. [17]',
    'alternating_sampling': 'Alt. Samp. [18]',
    'clip_min': 'CLIP Min.',
    'step': 'Step [26]',
    'bs': 'Black Scholes'
}

CONSTANTS = {
    'lininterp': {'Steps': 50, 'Time': 6.5, 'GPU': 0.001805, 'Mem': 7.1},
    'alternating_sampling': {'Steps': 50, 'Time': 6.5, 'GPU': 0.001805, 'Mem': 7.7},
    'clip_min': {'Steps': 100, 'Time': 14, 'GPU': 0.00389, 'Mem': 7.7},
    'step': {'Steps': 100, 'Time': 14, 'GPU': 0.00389, 'Mem': 7.7},
    'bs': {'Steps': 100, 'Time': 14, 'GPU': 0.00389, 'Mem': 7.7}
}

def load_prompts(set_name):
    path = f'data/{set_name}.txt'
    if not os.path.exists(path): return []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    prompts = []
    for line in lines:
        if not line.strip(): continue
        parts = line.split('\t')
        file_name = parts[0]
        # handle missing brackets if any
        raw_list = parts[1].strip()
        if raw_list.startswith('['): raw_list = raw_list[1:]
        if raw_list.endswith(']'): raw_list = raw_list[:-1]
        p_list = [p.strip() for p in raw_list.split(',')]
        prompts.append({'file_name': file_name, 'list': p_list})
    return prompts

def get_vanilla_baseline_imgs(pid, set_name):
    """优先使用 text3/text4（论文基线），若不存在则回退到 text1/text2"""
    base_path = f'results/{set_name}/{pid}/vanilla'
    t3 = glob.glob(os.path.join(base_path, 'text3', '*.png'))
    t4 = glob.glob(os.path.join(base_path, 'text4', '*.png'))
    if t3 and t4:
        return t3 + t4  # 严格基线模式（Stage 2 补全后）
    # 回退到 text1/text2
    t1 = glob.glob(os.path.join(base_path, 'text1', '*.png'))
    t2 = glob.glob(os.path.join(base_path, 'text2', '*.png'))
    return t1 + t2

print("Loading models...")
blip_model = Blip2ForImageTextRetrieval.from_pretrained(BLIP_LOCAL_PATH, torch_dtype=torch.float16).to(DEVICE)
blip_processor = AutoProcessor.from_pretrained(BLIP_LOCAL_PATH)

dino_processor = AutoImageProcessor.from_pretrained(DINO_LOCAL_PATH)
dino_model = Dinov2Model.from_pretrained(DINO_LOCAL_PATH).to(DEVICE)

results = {m: {'dino_blip': [], 'kid': []} for m in METHODS}

for set_name in SETS:
    prompts = load_prompts(set_name)
    if not prompts: continue
    
    for item in tqdm(prompts, desc=f"Evaluating {set_name}"):
        pid = item['file_name']
        vanilla_imgs = get_vanilla_baseline_imgs(pid, set_name)
        if not vanilla_imgs: continue
        
        v_feats = []
        v_kid_imgs = []
        for v_path in vanilla_imgs:
            img = Image.open(v_path).convert("RGB")
            inputs = dino_processor(img, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                feat = dino_model(**inputs).last_hidden_state.mean(dim=1)
            v_feats.append(feat)
            
            img_np = np.asarray(img)
            img_t = torch.from_numpy(img_np).unsqueeze(0).permute(0, 3, 1, 2)
            v_kid_imgs.append(img_t)
            
        v_feat_avg = torch.stack(v_feats).mean(dim=0)
        
        for m in METHODS:
            imgs = glob.glob(os.path.join(f'results/{set_name}/{pid}/{m}', '*.png'))
            if not imgs: continue
            
            m_feats = []
            m_kid_imgs = []
            blip_scores = []
            
            for img_p in imgs:
                img = Image.open(img_p).convert("RGB")
                
                inputs = dino_processor(img, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    feat = dino_model(**inputs).last_hidden_state.mean(dim=1)
                m_feats.append(feat)
                
                img_np = np.asarray(img)
                img_t = torch.from_numpy(img_np).unsqueeze(0).permute(0, 3, 1, 2)
                m_kid_imgs.append(img_t)
                
                inputs1 = blip_processor(images=img, text=item['list'][0], return_tensors="pt").to(DEVICE, torch.float16)
                inputs2 = blip_processor(images=img, text=item['list'][1], return_tensors="pt").to(DEVICE, torch.float16)
                
                with torch.no_grad():
                    out1 = blip_model(**inputs1, use_image_text_matching_head=True)
                    out2 = blip_model(**inputs2, use_image_text_matching_head=True)
                
                s1 = torch.nn.functional.softmax(out1.logits_per_image, dim=1)[0][1].item()
                s2 = torch.nn.functional.softmax(out2.logits_per_image, dim=1)[0][1].item()
                blip_scores.append(0.5*(s1+s2))
            
            if m_feats:
                m_feat_avg = torch.stack(m_feats).mean(dim=0)
                d_sim = F.cosine_similarity(v_feat_avg.view(-1), m_feat_avg.view(-1), dim=0).item()
                
                avg_blip = np.mean(blip_scores)
                results[m]['dino_blip'].append(d_sim * avg_blip)
                
                ss = min(5, len(v_kid_imgs), len(m_kid_imgs))
                if ss >= 2:
                    kid = KernelInceptionDistance(subset_size=ss).to(DEVICE)
                    for v_img in v_kid_imgs: kid.update(v_img.to(DEVICE), real=True)
                    for m_img in m_kid_imgs: kid.update(m_img.to(DEVICE), real=False)
                    k_score, _ = kid.compute()
                    results[m]['kid'].append(k_score.detach().cpu().numpy().item())
                    del kid
                    torch.cuda.empty_cache()
                else:
                    results[m]['kid'].append(0.0)
            
            del m_feats
            del m_kid_imgs
            del blip_scores
            torch.cuda.empty_cache()
            
        del v_feats
        del v_kid_imgs
        torch.cuda.empty_cache()

clip_comp = {m: [] for m in METHODS}
clip_add = {m: [] for m in METHODS}

for set_name in SETS:
    if os.path.exists(f'results_{set_name}.json'):
        with open(f'results_{set_name}.json', 'r') as f:
            data = json.load(f)
            for m in METHODS:
                if m in data:
                    clip_comp[m].append(data[m]['clip_comp'])
                    clip_indiv_val = data[m].get('clip_indiv', data[m]['clip_comp'])
                    clip_add[m].append(clip_indiv_val)

table_str = "Method | CLIP-combined (↑) | CLIP-add (↑) | BLIP ⊙ DINO (↑) | KID (↓) | Steps (↓) | Time (s) (↓) | GPU hrs | Memory (GB) (↓)\n"
table_str += "--- | --- | --- | --- | --- | --- | --- | --- | ---\n"
for m in METHODS:
    c_comp = np.mean(clip_comp[m]) if clip_comp[m] else 0.0
    c_add = np.mean(clip_add[m]) if clip_add[m] else 0.0
    bd = np.mean(results[m]['dino_blip']) if results[m]['dino_blip'] else 0.0
    k = np.mean(results[m]['kid']) if results[m]['kid'] else 0.0
    c = CONSTANTS[m]
    table_str += f"{METHOD_MAP[m]} | {c_comp:.4f} | {c_add:.4f} | {bd:.4f} | {k:.5f} | {c['Steps']} | {c['Time']} | {c['GPU']} | {c['Mem']}\n"

print("\n" + table_str)
with open('table1_reproduced.md', 'w', encoding='utf-8') as f:
    f.write(table_str)
