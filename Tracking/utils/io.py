import os
from typing import Dict
import numpy as np

from utils.log import logger


def write_results(filename, results_dict: Dict, data_type: str):
    if not filename:
        return
    path = os.path.dirname(filename)
    if not os.path.exists(path):
        os.makedirs(path)

    if data_type in ('mot', 'mcmot', 'lab', 'parcel'):
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, frame_data in results_dict.items():
            for tlwh, track_id in frame_data:
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, score=1.0)
                f.write(line)
    logger.info('Save results to {}'.format(filename))



def unzip_objs(objs):
    if len(objs) > 0:
        tlwhs, ids, scores = zip(*objs)
    else:
        tlwhs, ids, scores = [], [], []
    tlwhs = np.asarray(tlwhs, dtype=float).reshape(-1, 4)

    return tlwhs, ids, scores

def get_groudtruth(filename):
    with open(filename, 'r') as f:
        lines = f.read().split('\n')
        dic = dict()
        for line in lines:
            if len(line)>10:
                split = line.split(':')
                frame = get_frame(split[0], '.jpg')
                parcels = split[1].split('; ')
                elem_list = []
                for parcel in parcels:
                    if len(parcel)>4:
                        bbox_id = get_id_bbox(parcel)
                        dim = bbox_id[0]
                        if min(dim[2],dim[3])>30:
                            elem_list.append(bbox_id)
                dic[frame] = elem_list
    return dic



def get_frame(line, regex):
    split = line.split(regex)
    return int(split[0])



def get_id_bbox(element):
    if len(element)==0:
        return ()
    element = element.replace('[','').replace(']','').replace(',','').split(' ')
    c_id = int(element[0].split('.')[1])
    tly = int(element[1])
    tlx = int(element[2])
    w = int(element[3]) - int(element[1])
    h = int(element[4]) - int(element[2])
    return ((tly, tlx, w, h),c_id, 1)


def read_colis_results(filename):
    with open(filename) as f:
        lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i].replace(':\n','').replace(':', '; ').replace('; \n','')
        split = line.split('; ')[1:]
        to_add = []
        for parcel in split:
            parcel = parcel.split(' [')[1]
            parcel = parcel.replace(']','').split(', ')
            to_add_cor = []
            for j in parcel:
                to_add_cor.append(int(j))
            to_add_cor_2 = [to_add_cor[1], to_add_cor[0], to_add_cor[3]-to_add_cor[1], to_add_cor[2]-to_add_cor[0]]
            if min(to_add_cor_2[2], to_add_cor_2[3])>30:
                to_add.append(to_add_cor_2)
        lines[i] = to_add    
    return lines


def read_result_file(filename):
    results_dict = dict()
    with open(filename) as f:
         for line in f.readlines():
            linelist = line.split(',')
            fid = int(linelist[0])
            results_dict.setdefault(fid, list())
            score = 1
            tlwh = tuple(map(float, linelist[2:6]))
            target_id = int(linelist[1])

            results_dict[fid].append((tlwh, target_id, score))

    return results_dict


def read_gt_result(filename):
    with open(filename, 'r') as f:
        lines = f.read().split('\n')
        dic = dict()
        for line in lines:
            if len(line)>10:
                split = line.split(':')
                frame = get_frame(split[0], '.jpg')
                parcels = split[1].split('; ')
                elem_list = []
                for parcel in parcels:
                    if len(parcel)>4:
                        bbox_id = get_id_bbox_res(parcel)
                        dim = bbox_id[0]
                        if min(dim[2],dim[3])>30:
                            elem_list.append(bbox_id)
                dic[frame] = elem_list
    return dic


def get_id_bbox_res(element):
    if len(element)==0:
        return ()
    element = element.replace('[','').replace(']','').replace(',','').split(' ')
    c_id = int(element[0].split('.')[1])
    tly = int(element[1])
    tlx = int(element[2])
    w = int(element[3]) - int(element[1])
    h = int(element[4]) - int(element[2])
    return ((tlx, tly, w, h),c_id, 1)