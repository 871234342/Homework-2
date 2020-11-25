import torch
from torch.backends import cudnn

from backbone import EfficientDetBackbone
import numpy as np
import os
import json

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# load data
num_files = 0
for subdir, dirs, files in os.walk('test'):
    num_files = len(files)
    break

tags = np.linspace(1, num_files, num_files)
names = []
for tag in tags:
    names.append('test/' + str(int(tag)) + '.png')

# settings
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (1.5, 0.5)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
threshold = 0.2
iou_threshold = 0.2
compound_coef = 3
load_path = 'weights6.pth'
use_cuda = torch.cuda.is_available()
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True
force_input_size = 896

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

obj_list_num = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

# model
model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list_num),
                             ratios=anchor_ratios, scales=anchor_scales)
try:
    model.load_state_dict(torch.load(load_path, map_location='cpu'))
except RuntimeError as e:
    print(f'[Warning] Ignoring {e}')

model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()

# inferring
preds = []
for name in names:
    ori_imgs, framed_imgs, framed_metas = preprocess(name, max_size=input_size)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(x,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)

    out = invert_affine(framed_metas, out)

    o_bbox = out[0]['rois']
    o_cls = out[0]['class_ids']
    o_score = out[0]['scores']

    for i, ele in enumerate(o_bbox):
        o_bbox[i][0], o_bbox[i][1] = int(o_bbox[i][1]), int(o_bbox[i][0])
        o_bbox[i][2], o_bbox[i][3] = int(o_bbox[i][3]), int(o_bbox[i][2])

        o_cls[i] += 1

    pred = dict()
    pred['bbox'] = o_bbox.tolist()
    pred['score'] = o_score.tolist()
    pred['label'] = o_cls.tolist()
    preds.append(pred)


with open('predictions.json', 'w') as f:
    json.dump(preds, f)
