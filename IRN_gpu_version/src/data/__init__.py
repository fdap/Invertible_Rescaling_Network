'''create dataset and dataloader'''
import logging
import os
import mindspore
import mindspore.dataset as de

def create_dataset(phase, dataset_opt,gpu_ids, repeat_num=1):
    mode = dataset_opt['mode']
    if mode == 'LQGT':
        from src.data.LQGT_dataset import LQGTDataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))

    if phase == "train":
        dataset = D(dataset_opt)
        sampler = None
        num_workers = dataset_opt['n_workers'] * len(gpu_ids)
        shuffle = True
        de_dataset =  de.GeneratorDataset(dataset,["low_quality","ground_truth"],sampler=sampler,
                                    num_parallel_workers=num_workers,shuffle=shuffle)  

        batch_size = dataset_opt['batch_size']
        de_dataset=de_dataset.batch(batch_size,drop_remainder=True)

        columns_to_project = ["low_quality","ground_truth"]
        de_dataset = de_dataset.project(columns=columns_to_project)
        
        de_dataset = de_dataset.repeat(repeat_num)

        return de_dataset
    else:
        dataset = D(dataset_opt)
        sampler = None
        num_workers = 1
        shuffle = False
        de_dataset =  de.GeneratorDataset(dataset,["low_quality","ground_truth"],sampler=sampler,
                                    num_parallel_workers=num_workers,shuffle=shuffle) 

        batch_size = 1
        de_dataset=de_dataset.batch(batch_size)

        columns_to_project = ["low_quality","ground_truth"]
        de_dataset = de_dataset.project(columns=columns_to_project)

        return de_dataset






