from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks import CheckpointHook

@HOOKS.register_module()
class SaveBestCkptHook(CheckpointHook):
    def __init__(self,save_best_metrices,**kwargs):
        super(SaveBestCkptHook,self).__init__(**kwargs)
        self.save_best_metrices=save_best_metrices

    def after_train_epoch(self, runner):
        super(SaveBestCkptHook,self).after_train_epoch(runner)
        for metric in self.save_best_metrices:
            if 'best_val_metircs' in runner.meta:
                if metric in runner.meta['best_val_metircs']:
                    if runner.meta['best_val_metircs'][metric][1]==runner.epoch:
                        runner.save_checkpoint(self.out_dir,filename_tmpl='best_%s=%.3f_at_epoch%d.pth'%
                                      (metric,runner.meta['best_val_metircs'][metric][0],runner.epoch),save_optimizer=False,create_symlink=False)
                        runner.logger.info('New Best %s=%f model saved.'%(metric,runner.meta['best_val_metircs'][metric][0]))