# -*- coding: utf-8 -*-


from eval_mot import main
import sys


if __name__ == '__main__':
    seqs_str = '''
    CAM00
    '''
    
    seq_file = 'Tracking/sequences.txt'
    with open(seq_file, 'rb') as f:
        text = f.read().decode('utf16')
    lines = text.split('\n')
    root = lines[0]
    seq = []
    for i in range(1, len(lines)):
        seq.append(lines[i])
        
    seqs = [seq.strip() for seq in seqs_str.split()]
    
    _, show_image, save_image, Eval = sys.argv
    
    main(data_root= root,
         seqs=seqs,
         exp_name='colis',
         show_image = show_image,
         save_image = save_image,
         Eval = Eval)
