import time
from scipy.optimize import linear_sum_assignment
import torch.nn as nn
import numpy as np
from skimage.io import imsave
from collections import defaultdict
import pandas as pd
import os
from clize import run
from dataset import COCO
from dataset import VOC
from dataset import SubSample
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.nn.functional import smooth_l1_loss, cross_entropy, mse_loss, l1_loss
from model import SeqDetect
from torchvision.models import vgg16
from torchvision import models
import cv2
from torchvision.datasets.folder import default_loader

cudnn.benchmark = True


def train(*, config='config', resume=False):
    cfg = _read_config(config)
    batch_size = cfg['batch_size']
    num_epoch = cfg['num_epoch']
    debug = cfg['debug']
    out_folder = cfg['out_folder']
    (train_dataset, valid_dataset), (train_evaluation, valid_evaluation) = _build_dataset(cfg)
    
    print('Done loading dataset annotations.')
    if debug:
        n = 16 
        train_dataset = SubSample(train_dataset, nb=n)
        valid_dataset = SubSample(valid_dataset, nb=n)
        train_evaluation = SubSample(train_evaluation, nb=n)
        valid_evaluation = SubSample(valid_evaluation, nb=n)
    assert train_dataset.class_to_idx == valid_dataset.class_to_idx
    assert train_dataset.idx_to_class == valid_dataset.idx_to_class
    clfn = lambda l:l
    # Dataset loaders for full training and full validation 
    train_loader = DataLoader(
        train_dataset, 
        shuffle=True, 
        batch_size=batch_size, 
        collate_fn=clfn,
        num_workers=cfg['num_workers'],
    )
    train_evaluation_loader = DataLoader(
        train_evaluation,
        batch_size=batch_size,
        collate_fn=clfn,
        num_workers=cfg['num_workers'],
    )
    valid_evaluation_loader = DataLoader(
        valid_evaluation,
        batch_size=batch_size,
        collate_fn=clfn,
        num_workers=cfg['num_workers'],
    )
    class_ids = list(set(train_dataset.class_to_idx.values()))
    class_ids = sorted(class_ids)
    nb_classes = len(class_ids)
    print('Number of training images : {}'.format(len(train_dataset)))
    print('Number of valid images : {}'.format(len(valid_dataset)))
    print('Number of classes : {}'.format(nb_classes))
    stats_filename = os.path.join(out_folder, 'stats.csv')
    train_stats_filename = os.path.join(out_folder, 'train_stats.csv')
    model_filename = os.path.join(out_folder, 'model.th')

    if resume:
        model = torch.load(model_filename)
        model = model.cuda()
        if os.path.exists(stats_filename):
            stats = pd.read_csv(stats_filename).to_dict(orient='list')
            first_epoch = max(stats['epoch']) + 1
        else:
            stats = defaultdict(list)
            first_epoch = 0
        if os.path.exists(train_stats_filename):
            train_stats = pd.read_csv(train_stats_filename).to_dict(orient='list')
        else:
            train_stats = defaultdict(list)
    else:
        nb_steps = cfg['nb_steps']
        hidden_size = cfg['hidden_size']
        base = cfg['base']
        if base == 'vgg16':
            base = vgg16(pretrained=True).features.cuda()
            input_size = 512
        elif base.startswith('resnet'):
            model_class = getattr(models, base)
            resnet = model_class(pretrained=True)
            base = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4
            )
            input_size = 512
        elif base == None:
            input_size = 3 * cfg['image_size'] * cfg['image_size']
        else:
            raise ValueError(base)
        output_size = 4 + nb_classes
        model = SeqDetect(
            input_size, output_size, 
            base=base, 
            nb_steps=nb_steps, 
            hidden_size=hidden_size)
        model = model.cuda()
        model.transform = valid_dataset.transform 
        model.nb_classes = nb_classes
        first_epoch = 0
        stats = defaultdict(list)
        train_stats = defaultdict(list)
        model.nb_updates = 0
        model.avg_loss = 0.
        model.avg_loc = 0.
        model.avg_classif = 0
        model.class_to_idx = train_dataset.class_to_idx
        model.idx_to_class = train_dataset.idx_to_class
        model.cfg = cfg
        print(model)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(first_epoch, num_epoch):
        model.train()
        for batch, samples, in enumerate(train_loader):
            t0 = time.time()
            X, tb, tc, pb, pc, ob, oc = _predict(model, samples)
            model.zero_grad()
            mask = (oc!=0).view(oc.size(0), oc.size(1), 1).float()
            l_loc = (mask * smooth_l1_loss(pb, ob, size_average=False, reduce=False)).sum() / mask.sum()
            pc_ = pc.view(pc.size(0) * pc.size(1), pc.size(2))
            oc_ = oc.view(-1)
            l_classif = cross_entropy(pc_, oc_)
            loss = cfg['lambda_'] * l_loc + l_classif
            loss.backward()
            _update_lr(optimizer, model.nb_updates, cfg['lr_schedule'])
            optimizer.step()
            delta = time.time() - t0

            print('Epoch {:05d}/{:05d} Batch {:05d}/{:05d} Loss : {:.3f} Loc : {:.3f} '
                  'Classif : {:.3f} Time:{:.3f}s'.format(
                      epoch + 1,
                      num_epoch,
                      batch + 1, 
                      len(train_loader), 
                      loss.data[0], 
                      l_loc.data[0],
                      l_classif.data[0],
                      delta
                    ))
            train_stats['loss'].append(loss.data[0])
            train_stats['loc'].append(l_loc.data[0])
            train_stats['classif'].append(l_classif.data[0])
            train_stats['time'].append(delta)
            if model.nb_updates % cfg['log_interval'] == 0:
                pd.DataFrame(train_stats).to_csv(train_stats_filename, index=False)
                t0 = time.time()
                torch.save(model, model_filename)
                ob = ob.data.cpu().numpy() * cfg['image_size']
                pb = pb.data.cpu().numpy() * cfg['image_size']
                tb = tb * cfg['image_size']
                pc = pc.data.cpu().numpy()
                oc = oc.data.cpu().numpy()
                
                X = X.data.cpu().numpy()
 
                for i in range(len(X)):
                    x = X[i]
                    x = x.transpose((1, 2, 0))
                    x = x * np.array(cfg['std']) + np.array(cfg['mean'])
                    x = x.astype('float32')
                    gt_boxes = tb[i].tolist()
                    gt_labels = tc[i].tolist()
                    gt_boxes = [(b, train_dataset.idx_to_class[l]) for b, l in zip(gt_boxes, gt_labels) if l != 0]
                    pred_boxes = pb[i].tolist()
                    pred_labels = pc[i].argmax(axis=1).tolist()
                    pred_boxes = [(b, train_dataset.idx_to_class[l]) for b, l in zip(pred_boxes, pred_labels) if l != 0]
                    pad = cfg['pad']
                    im = np.zeros((x.shape[0] + pad * 2, x.shape[1] + pad * 2, x.shape[2]))
                    im[pad:-pad, pad:-pad] = x
                    im = _draw_bounding_boxes(im, gt_boxes, color=(1, 0, 0), text_color=(1, 0, 0), pad=pad)
                    im = _draw_bounding_boxes(im, pred_boxes, color=(0, 1, 0), text_color=(0, 1, 0), pad=pad) 
                    imsave(os.path.join(out_folder, 'train/sample_{:05d}.jpg'.format(i)), im)
            model.nb_updates += 1
        if debug and model.nb_updates % 50 != 0:
            continue
        if epoch % cfg['eval_interval'] != 0:
            continue
        metrics = defaultdict(list)
        print('Evaluation')
        for (split_name, loader) in (('train', train_evaluation_loader), ('valid', valid_evaluation_loader)):
            t0 = time.time()
            ex = 0
            for batch, samples, in enumerate(loader):
                t0 = time.time()
                X, tb, tc, pb, pc, ob, oc = _predict(model, samples) 
                ob = ob.data.cpu().numpy() * cfg['image_size']
                pb = pb.data.cpu().numpy() * cfg['image_size']
                tb = tb * cfg['image_size']
                pc = pc.data.cpu().numpy()
                oc = oc.data.cpu().numpy()
                
                X = X.data.cpu().numpy()
 
                for i in range(len(X)):
                    x = X[i]
                    x = x.transpose((1, 2, 0))
                    x = x * np.array(cfg['std']) + np.array(cfg['mean'])
                    x = x.astype('float32')
                    gt_boxes = tb[i].tolist()
                    gt_labels = tc[i].tolist()
                    gt_boxes = [(b, train_dataset.idx_to_class[l]) for b, l in zip(gt_boxes, gt_labels) if l != 0]
                    pred_boxes = pb[i].tolist()
                    pred_labels = pc[i].argmax(axis=1).tolist()
                    pred_boxes = [(b, train_dataset.idx_to_class[l]) for b, l in zip(pred_boxes, pred_labels) if l != 0]
                    for class_id, class_name in train_dataset.idx_to_class.items():
                        if class_id == 0:
                            continue
                        p = [b for b, c in pred_boxes if c == class_name]
                        t = [b for b, c in gt_boxes if c == class_name]
                        if len(p) == 0 or len(t) == 0:
                            precision = 0
                            recall = 0
                        else:
                            p = torch.Tensor([p])
                            t = torch.Tensor([t])
                            [m] = _iou(p, t) > cfg['iou_threshold']
                            precision = (m.sum(1) > 0).float().mean()
                            recall = (m.sum(0) > 0).float().mean()
                        metrics['precision_' + class_name + '_' + split_name].append(precision)
                        metrics['recall_' + class_name + '_' + split_name].append(recall)
                    pad = cfg['pad']
                    im = np.zeros((x.shape[0] + pad * 2, x.shape[1] + pad * 2, x.shape[2]))
                    im[pad:-pad, pad:-pad] = x
                    im = _draw_bounding_boxes(im, gt_boxes, color=(1, 0, 0), text_color=(1, 0, 0), pad=pad)
                    im = _draw_bounding_boxes(im, pred_boxes, color=(0, 1, 0), text_color=(0, 1, 0), pad=pad) 
                    imsave(os.path.join(out_folder, 'eval_{}/sample_{:05d}.jpg'.format(split_name, ex)), im)
                    ex += 1

                delta = time.time() - t0
                print('Eval Batch {:04d}/{:04d} on split {} Time : {:.3f}s'.format(batch, len(loader), split_name, delta))

        for k in sorted(metrics.keys()):
            v = np.mean(metrics[k])
            print('{}: {:.4}'.format(k, v))
            stats[k].append(v)
        stats['epoch'].append(epoch)
        pd.DataFrame(stats).to_csv(stats_filename, index=False)


def _predict(model, samples):
    X = torch.stack([x for x, _, _ in samples], 0) 
    X = X.cuda()
    X = Variable(X)
    # X has shape (nb_examples, 3, image_size, image_size)
    ypred = model(X)
    # ypred has shape (nb_examples, nb_steps, 4+nb_classes)

    tb = [_pad([b for b, cl in bb], model.nb_steps, size=4) for (_, _, bb) in samples]
    tb = torch.Tensor(tb)
    # tb = true boxes
    # it has shape (nb_examples, nb_steps, 4) with 0 on all entries for non-boxes
    tb = tb.float()
     
    tc = [_pad([cl for b, cl in bb], model.nb_steps, size=0) for (_, _, bb) in samples]
    # tc = true classes
    # it has shape (nb_examples, nb_steps) with 0 (background) for non-boxes
    tc = torch.Tensor(tc)
    tc = tc.long()
    
    pb = ypred[:, :, 0:4].contiguous()
    pc = ypred[:, :, 4:].contiguous()
    # pb = pred boxes
    # pc = pred classes

    pb_ = pb.cpu().data
    pc_ = pc.cpu().data
        
    # match prediction steps with true boxes and put
    # the result in ob, oc, and om
    center_in = _center_in(pb_, tb).float()
    iou = _iou(pb_, tb).float()
    dist = _l1_dist(pb_, tb).float()
    #ob, oc = build_out_hungarian(pb_, pc_, tb, tc, center_in, iou, dist)
    ob, oc = build_out_fixed(pb_, pc_, tb, tc)
    # ob : output boxes for each prediction step of the rnn
    # oc : output classes for each prediction step of the rnn
    # om : output masks for each prediction step of the rnn
    ob = Variable(ob).cuda()
    oc = Variable(oc).cuda()
    return X, tb, tc, pb, pc, ob, oc

def build_out_fixed(pb, pc, tb, tc):
    return tb, tc

def build_out_hungarian(pb, pc, tb, tc, center_in, iou, dist):
    # pb = predicted boxes, shape (nb_examples, nb_preds, 4)
    # pc = predicted classes, shape (nb_examples, nb_preds, nb_classes)
    # tb =  true boxes, shape (nb_examples, nb_true, 4)
    # tc = true classes, shape (nb_examples, nb_true)
    nb = len(pb)
    nbp = pb.size(1)
    nbt = tb.size(1)

    ob = torch.zeros_like(pb)
    oc = torch.zeros(pb.size(0), pb.size(1)).long()
    
    center_in = center_in.numpy()
    rank = torch.linspace(0, 1, nbp).view(-1, 1).repeat(1, nbt).numpy()
    dist = dist.numpy()

    for i in range(nb):
        cost = .6 * (1 - center_in[i]) + .3 * rank + .1 * dist[i]
        #cost[:, tc[i]==0] = 1
        _, col_ind = linear_sum_assignment(cost)
        col_ind = col_ind.tolist()
        ob[i] = tb[i][col_ind]
        oc[i] = tc[i][col_ind]
    # ob = output boxes for each rnn step, shape (nb_examples, nb_preds, 4)
    # oc = output classes for each rnn step, shape (nb_examples, nb_preds)
    return ob, oc


def _pad(l, nb, size=4):
    if len(l) > nb:
        print('len(l)>nb')
        l = l[0:nb]
    if size == 0:
        empty = 0
    else:
        empty = [0] * size
    return l + [empty] * (nb - len(l))


def _iou(pred_boxes, gt_boxes):
    # pred_boxes: (nb_examples, nbp, 4)
    # gt_boxes: (nb_examples, nbg, 4)
    p = pred_boxes
    t = gt_boxes

    p = p.view(p.size(0), p.size(1), 1, 4)
    px, py, pw, ph = p[:, :, :, 0], p[:, :, :, 1], p[:, :, :, 2], p[:, :, :, 3]

    t = t.view(t.size(0), 1, t.size(1), 4)
    tx, ty, tw, th = t[:, :, :, 0], t[:, :, :, 1], t[:, :, :, 2], t[:, :, :, 3]
    winter = torch.min(px + pw, tx + tw) - torch.max(px, tx) 
    hinter = torch.min(py + ph, ty + th) - torch.max(py, ty) 
    winter = winter.clamp_(min=0)
    hinter = hinter.clamp_(min=0)
    inter = winter * hinter
    union = pw * ph + tw * th - inter
    res = inter / union
    # res : (nb_examples, nbp, nbg)
    return res

def _center_in(pred_boxes, gt_boxes):
    p = pred_boxes
    t = gt_boxes
    p = p.view(p.size(0), p.size(1), 1, 4)
    px, py, pw, ph = p[:, :, :, 0], p[:, :, :, 1], p[:, :, :, 2], p[:, :, :, 3]
    cpx = px + pw / 2
    cpy = py + ph / 2

    t = t.view(t.size(0), 1, t.size(1), 4)
    tx, ty, tw, th = t[:, :, :, 0], t[:, :, :, 1], t[:, :, :, 2], t[:, :, :, 3]
    
    return (cpx >= tx) * (cpx <= tx + tw) * (cpy >= ty) * (cpy <= ty + th)

def _l1_dist(pred_boxes, gt_boxes):
    p = pred_boxes
    t = gt_boxes
    p = p.view(p.size(0), p.size(1), 1, 4)
    t = t.view(t.size(0), 1, t.size(1), 4)
    return torch.abs(p - t).sum(3)

    
def _build_dataset(cfg):
    mean = cfg['mean']
    std = cfg['std']
    image_size = cfg['image_size']
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    dataset = cfg['dataset']
    if dataset == 'COCO':
        train_split = 'train' + str(cfg['dataset_version'])
        val_split = 'val' + str(cfg['dataset_version'])
        kwargs = dict(
            folder=cfg['dataset_root_folder'],
            classes=cfg['classes'],
            transform=train_transform,
            image_size=cfg['image_size'],
        )
        train_dataset = COCO(
            split=train_split,
            data_augmentation_params=cfg['data_augmentation_params'],
            **kwargs
        )
        valid_dataset = COCO(
            split=val_split,
            data_augmentation_params={},
            **kwargs,
        )
        train_evaluation = SubSample(COCO(
            split=train_split,
            data_augmentation_params={},
            **kwargs,
        ), nb=cfg['train_evaluation_size'])
        valid_evaluation = SubSample(
            valid_dataset, 
            nb=cfg['val_evaluation_size']
        )
    elif dataset == 'VOC':
        kwargs = dict(
            folder=cfg['dataset_root_folder'],
            which=cfg['dataset_version'], 
            classes=cfg['classes'],
            transform=train_transform,
            image_size=cfg['image_size']
        )
        train_dataset = VOC(
            split='train',
            data_augmentation_params=cfg['data_augmentation_params'],
            **kwargs,
        )
        valid_dataset = VOC(
            split='val',
            data_augmentation_params={},
            **kwargs,
        )
        train_evaluation = SubSample(VOC(
            split='train',
            data_augmentation_params={},
            **kwargs
        ), nb=cfg['train_evaluation_size'])
        valid_evaluation = SubSample(
            valid_dataset, 
            nb=cfg['val_evaluation_size']
        )
    else:
        raise ValueError('Unknown dataset {}'.format(dataset))
    return (train_dataset, valid_dataset), (train_evaluation, valid_evaluation)

def _update_lr(optimizer, nb_iter, schedule):
    for sc in schedule:
        (start_iter, end_iter), new_lr = sc['iter'], sc['lr']
        if start_iter <= nb_iter  <= end_iter:
            break
    old_lr = optimizer.param_groups[0]['lr']
    if old_lr != new_lr:
        print('Chaning LR from {:.5f} to {:.5f}'.format(old_lr, new_lr))
    for g in optimizer.param_groups:
        g['lr'] = new_lr



def _read_config(config):
    cfg = {}
    exec(open(config).read(), {}, cfg)
    return cfg


def test(filename, *, model='out/model.th', nb_predictions=10, out=None, cuda=False):
    if not out:
        path, ext = filename.split('.', 2)
        out = path + '_out' + '.' + ext
    model = torch.load(model, map_location=lambda storage, loc: storage)
    model.nb_steps = nb_predictions
    model.use_cuda = cuda
    if cuda:
        model = model.cuda()
    
    model.eval()
    im = default_loader(filename)
    x = model.transform(im)
    X = x.view(1, x.size(0), x.size(1), x.size(2))
    if cuda:
        X = X.cuda()
    X = Variable(X)
    ypred = model(X)
    pb = ypred[:, :, 0:4].contiguous()
    pc = ypred[:, :, 4:].contiguous()
    pb = pb.data.cpu().numpy() * model.cfg['image_size']
    pc = pc.data.cpu().numpy()
    X = X.data.cpu().numpy()
    x = X[0]
    x = x.transpose((1, 2, 0))
    x = x * np.array(model.cfg['std']) + np.array(model.cfg['mean'])
    x = x.astype('float32')
    pred_boxes = pb[0].tolist()
    pred_labels = pc[0].argmax(axis=1).tolist()
    pred_scores = softmax(pc[0], axis=1).max(axis=1).tolist()
    
    for box, cl, score in zip(pred_boxes, pred_labels, pred_scores):
        bx, by, bw, bh = box
        b = int(bx), int(by), int(bx + bw), int(by + bh)
        print('{:02d},{:02d},{:02d},{:02d} {} {:.2f}'.format(*b, model.idx_to_class[cl], score))
    
    pred_boxes = [(b, model.idx_to_class[l]) for b, l in zip(pred_boxes, pred_labels) if l != 0]
    
    pad = model.cfg['pad']
    im = np.zeros((x.shape[0] + pad * 2, x.shape[1] + pad * 2, x.shape[2]))
    im[pad:-pad, pad:-pad] = x
    im = _draw_bounding_boxes(im, pred_boxes, color=(0, 1, 0), text_color=(0, 1, 0), pad=pad) 
    imsave(out, im)

def softmax(x, axis=1):
    e_x = np.exp(x - x.max(axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True) # only difference"




def _draw_bounding_boxes(
    image,
    bbox_list, 
    color=[1.0, 1.0, 1.0], 
    text_color=(1, 1, 1), 
    font=cv2.FONT_HERSHEY_PLAIN, 
    font_scale=1.0,
    pad=0):
    for bbox, class_name in bbox_list:
        x, y, w, h = bbox
        x = int(x) + pad
        y = int(y) + pad
        w = int(w)
        h = int(h)
        if x > image.shape[1]:
            continue
        if x + w > image.shape[1]:
            continue
        if y > image.shape[0]:
            continue
        if y + h > image.shape[1]:
            continue
        image = cv2.rectangle(image, (x, y), (x + w, y + h), color)
        text = class_name
        image = cv2.putText(image, text, (x, y), font, font_scale, text_color, 2, cv2.LINE_AA)
    return image

if __name__ == '__main__':
    run([train, test])
