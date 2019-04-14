import os
import cv2
import logging
import motmetrics as mm
from tracker.mot_tracker import OnlineTracker

from datasets.colis_seq import get_loader
from utils import visualization as vis
from utils.log import logger
from utils.timer import Timer
import numpy as np
from utils.evaluation import Evaluator
import time
import matplotlib.pyplot as plt
import sys

def mkdirs(path):
    if os.path.exists(path):
        return
    os.makedirs(path)


def write_results(filename, results, data_type):
    save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def eval_seq(dataloader, data_type, result_filename, save_dir=None, show_image=True):
    if save_dir is not None:
        mkdirs(save_dir)

    tracker = OnlineTracker(os.getcwd()+'/MOTDTwVideo/data/colis_google.pth')
    timer = Timer()
    op_time = []
    results = []
    wait_time = 1
    fps, frame_count = time.time(), 0
    frame_count = 0
    for frame_id, batch in enumerate(dataloader):
        if frame_count % 20 == 0:
            logger.info('Processing frame {}'.format(frame_id))

        frame, det_tlwhs, det_scores, _, _ = batch

        # run tracking
        online_targets, timing = tracker.update(frame, det_tlwhs, None)
        op_time.append(timing)
        
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            online_tlwhs.append(t.tlwh)
            online_ids.append(t.track_id)
        timer.toc()

        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))

        if show_image == True or save_dir is not None:
            online_im = vis.plot_tracking(frame, online_tlwhs, online_ids, frame_id=frame_id,
                                      fps=1. / (timer.average_time+1))
        if show_image:
            cv2.imshow('online_im', online_im)
            key = cv2.waitKey(wait_time)
            key = chr(key % 128).lower()
            if key == 'q':
                exit(0)
            elif key == 'p':
                cv2.waitKey(0)
            elif key == 'a':
                wait_time = int(not wait_time)
        if save_dir is not None:
            plt.imsave(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im[:,:,[2,1,0]])
        frame_count += 1

    fps = time.time()-fps
    # save results
    analyse_time(op_time, fps, frame_count)
    write_results(result_filename, results, data_type)
    
    
    
def analyse_time(timer, duration, frame_count):
    print('------ Sequence Information -----')
    duration = max(duration, 1)
    fps = frame_count / duration
    timer = np.transpose((np.array(timer)))

    Names = ["Avg Prediction time:","Avg Scoring time:","Avg Association time:","Avg New stracks:","Avg Update state:"]
    avg = [round(np.mean(timer[i]),5) for i in range(len(timer))]
    print('fps rate: {} s'.format(fps))
    for i in range(len(Names)):
        print(Names[i], avg[i])

    
    

def main(data_root='/data', det_root=None,
         seqs=('MOT16-05',), exp_name='demo', save_image=False, show_image=True, Eval = True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdirs(result_root)
    data_type = 'mot'

    # run tracking
    accs = []
    for seq in seqs:
        output_dir = os.path.join(data_root, 'outputs', seq) if save_image else None

        logger.info('start seq: {}'.format(seq))
        loader = get_loader(data_root, det_root, seq)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        eval_seq(loader, data_type, result_filename,
                 save_dir=output_dir, show_image=show_image)
        

        # eval
        if Eval == True:
            logger.info('Evaluate seq: {}'.format(seq))
            evaluator = Evaluator(data_root, seq, data_type)
            accs.append(evaluator.eval_file(result_filename))

    # get summary
    if Eval == True:
        # metrics = ['mota', 'num_switches', 'idp', 'idr', 'idf1', 'precision', 'recall']
        metrics = mm.metrics.motchallenge_metrics
        # metrics = None
        mh = mm.metrics.create()
        summary = Evaluator.get_summary(accs, seqs, metrics)
        strsummary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names
        )
        print(strsummary)
        Evaluator.save_summary(summary, os.path.join(result_root, f'summary_{exp_name}.xlsx'))


if __name__ == '__main__':

    seq_file = 'MOTDTwVideo/sequences.txt'
    with open(seq_file, 'rb') as f:
        text = f.read().decode('utf8')
    lines = text.split('\n')
    root = lines[1][:-1]
    seq = []
    for i in range(3, len(lines)):
        seq.append(lines[i])
    
    _, show_image, save_image, Eval = sys.argv

    main(data_root= root,
         seqs=seq,
         exp_name='colis',
         show_image = int(show_image),
         save_image = int(save_image),
         Eval = int(Eval))