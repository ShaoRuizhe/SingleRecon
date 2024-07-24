# encoding:utf-8
import os
import os.path as osp
from collections import OrderedDict

from mmcv.runner import Hook
from torch.utils.data import DataLoader
from mmcv.runner.hooks import HOOKS


@HOOKS.register_module()
class EvalHook(Hook):
    """Evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
    """

    def __init__(self, dataloader, interval=1, show=False, show_interval=200, show_score_thr=0.5, **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got'
                            f' {type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.eval_kwargs = eval_kwargs
        self.show = show
        if self.show:
            self.show_interval = show_interval
            self.show_score_thr = show_score_thr
        else:
            self.show_interval = None
            self.show_score_thr = None

    def after_train_epoch(self, runner):
        if isinstance(self.interval,int):
            if not self.every_n_epochs(runner, self.interval):
                return
        elif isinstance(self.interval,dict):
            stage_num=0
            cur_interval=0
            for i,stage in enumerate(self.interval['stage']):
                if runner.epoch+1<stage:
                    break
                stage_num=i
                cur_interval=self.interval['interval'][stage_num]
            if cur_interval==0 or not (runner.epoch + 1-self.interval['stage'][stage_num]) % cur_interval == 0:
                return
        from mmdet.apis import single_gpu_test
        results = single_gpu_test(runner.model, self.dataloader, show=self.show,
                                  out_dir=os.path.join(runner.work_dir, 'images', 'ep' + str(runner.epoch)),
                                  show_score_thr=self.show_score_thr, show_interval=self.show_interval)
        # import numpy as np
        # runner.model.module.show_result(img=np.zeros((1, 3, 1024, 1024)), result=results[0])# roof必须是mask形式，模型的输出是一串不知所谓的二进制串
        self.evaluate(runner, results)

    def evaluate(self, runner, results):
        eval_res, _, _ = self.dataloader.dataset.evaluate(  # 第二项是dt_match匹配关系
            results, logger=runner.logger, **self.eval_kwargs)
        if 'best_val_metircs' in runner.meta:
            for name, val in eval_res.items():
                if 'mae' in name or 'polis' in name:  # todo: 学习vitae的调节方式
                    if not name in runner.meta['best_val_metircs'] or runner.meta['best_val_metircs'][name][0] > val:
                        runner.meta['best_val_metircs'][name] = [val, runner.epoch]
                elif 'iou' in name or 'mAP' in name or 'mean_cos_angle_error' in name:
                    if not name in runner.meta['best_val_metircs'] or runner.meta['best_val_metircs'][name][0] < val:
                        runner.meta['best_val_metircs'][name] = [val, runner.epoch]
        else:
            val_metric_dict = {}
            for name, val in eval_res.items():
                val_metric_dict[name] = [val, runner.epoch]
            runner.meta['best_val_metircs'] = val_metric_dict
        runner.log_buffer.output = OrderedDict(eval_res)
        # runner.log_buffer.update(log_eval, dict(zip(log_eval.keys(), [1]*len(log_eval))))
        runner.log_buffer.ready = True


class DistEvalHook(EvalHook):
    """Distributed evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
    """

    def __init__(self,
                 dataloader,
                 interval=1,
                 gpu_collect=False,
                 **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got '
                            f'{type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.gpu_collect = gpu_collect
        self.eval_kwargs = eval_kwargs

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        from mmdet.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)
