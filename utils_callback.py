import json
import os
import sys

from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm
from pytorch_lightning.callbacks import Callback, TQDMProgressBar, ModelCheckpoint
import matplotlib.pyplot as plt

class CustomFilenameCallback(Callback):
    # 根据测试集的指标修改保存文件夹的名字和保存测试指标到json文件中
    def on_test_end(self, trainer, pl_module):
        # 获取测试集指标信息
        test_metrics = trainer.callback_metrics  # 获取测试集指标
        main_metrice = test_metrics[f'test_{trainer.model.main_metrice}_epoch']
        # 修改文件名，将测试集损失信息添加到文件名中
        metrics = {key: float(value) for key, value in test_metrics.items()}
        original_dir_name = trainer.logger.log_dir
        components = original_dir_name.split("/")
        last_components = components[-1].split('\\')
        components = components[:-1] + last_components
        # new_path = os.path.join("/".join(components[:-1]), f"{trainer.model.main_metrice}:{main_metrice :.4f}")
        new_path = os.path.join("/".join(components[:-1]), f"{trainer.model.main_metrice}_{main_metrice :.4f}")
        new_path = new_path.replace('.', '_')
        f = open(os.path.join(trainer.logger.log_dir, 'metrics.json'), 'w')
        json.dump(metrics, f)
        f.close()

        def rename_folder(folder_path):
            # 检查文件夹是否存在
            if not os.path.exists(folder_path):
                return folder_path
            else:
                # 文件夹已存在，尝试在文件夹名字后添加序号
                base_folder_path = folder_path
                index = 1
                while os.path.exists(folder_path):
                    folder_path = f"{base_folder_path}({index})"
                    index += 1
                return folder_path
        os.rename(original_dir_name, rename_folder(new_path))  # 重命名文件夹或文件路径


class FineTune_ModelCheckpoint(ModelCheckpoint):
    def __init__(self, dirpath, save_top_k, monitor='val_loss', mode='min', filename='model-{epoch:02d}-{val_loss:.2f}',
                 save_last=False):
        super().__init__(dirpath=dirpath, save_top_k=save_top_k, monitor=monitor, mode=mode, filename=filename,
                         save_last=save_last)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # 获取非冻结参数的状态字典
        trainable_state_dict = {}
        for name, param in pl_module.named_parameters():
            #print(name)
            if param.requires_grad:
                trainable_state_dict[name] = param.data
        #trainer.callback_metrics
        # 更新检查点的状态字典
        checkpoint['state_dict']= trainable_state_dict
        # 调用父类的方法保存检查点
        super().on_save_checkpoint(trainer, pl_module, checkpoint)
        test_metrics = trainer.callback_metrics  # 获取测试集指标
        main_metrice = test_metrics[f'val_{trainer.model.main_metrice}_epoch']
        # 修改文件名，将测试集损失信息添加到文件名中
        metrics = {key: float(value) for key, value in test_metrics.items() if 'val' in key}
        original_dir_name = trainer.logger.log_dir
        components = original_dir_name.split("/")
        last_components = components[-1].split('\\')
        components = components[:-1] + last_components
        # new_path = os.path.join("/".join(components[:-1]), f"{trainer.model.main_metrice}:{main_metrice :.4f}")
        new_path = os.path.join("/".join(components[:-1]), f"{trainer.model.main_metrice}_{main_metrice :.4f}")
        f = open(os.path.join(trainer.logger.log_dir, 'metrics.json'), 'w')
        json.dump(metrics, f)
        f.close()


class ProgressBar(TQDMProgressBar):
    """Global progress bar.
    TODO: add progress bar for training, validation and testing loop.
    """
    def init_sanity_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for the validation sanity run."""
        return Tqdm(
            desc=self.sanity_check_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
        )

    def init_train_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for training."""
        return Tqdm(
            ncols = 200,
            desc=self.train_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=False,
            file=sys.stdout,
            smoothing=0,
        )

    def init_predict_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for predicting."""
        return Tqdm(
            ncols=200,
            desc=self.predict_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=False,
            file=sys.stdout,
            smoothing=0,
        )

    def init_validation_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for validation."""
        # The train progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.trainer.state.fn != "validate"
        return Tqdm(
            ncols=200,
            desc=self.validation_description,
            position=(2 * self.process_position + has_main_bar),
            disable=self.is_disabled,
            leave=not has_main_bar,
            dynamic_ncols=False,
            file=sys.stdout,
        )

    def init_test_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for testing."""
        return Tqdm(
            ncols=200,
            desc="Testing",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=False,
            file=sys.stdout,
        )


# class PlotCallback(Callback):
#     #https://lightning.ai/docs/torchmetrics/stable/pages/plotting.html
#     def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
#         index2mode = ['train', 'val', 'test']
#         # test_metrics = trainer.callback_metrics  # 获取测试集指标
#         # main_metrice = test_metrics[f'test_{trainer.model.main_metrice}_epoch']
#         trainer.model.metrics[trainer.model.main_metrice][index2mode.index('test')].plot()
#         plt.show()