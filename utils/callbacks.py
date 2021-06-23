import tensorflow as tf

class SaveCustomCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_manager, steps_per_epoch):
        self.checkpoint_manager = checkpoint_manager
        self.steps_per_epoch = steps_per_epoch
    

    def on_epoch_end(self, epoch, logs=None):
        self.checkpoint_manager.checkpoint.epoch.assign_add(1)
        self.checkpoint_manager.checkpoint.step.assign_add(self.steps_per_epoch)
        self.checkpoint_manager.checkpoint.psnr.assign(logs["val_psnr_metric"])
        self.checkpoint_manager.save()