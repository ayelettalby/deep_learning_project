import os
import time
from shutil import copy
#from SegmentationModule.Logger import LoggerHandler


"""
the following code is to define the model settings from a dictionary of settings which include the following keys:
-- dataset_settings
-- pre_processing_settings
-- dataloader_settings
-- augmentations_settings
-- compilation_settings
-- metrics_settings
-- output_settings
-- architecture_settings
-- training_settings
"""

class SegSettings(object):
    """
    this class is to define the Hydra project settings
    """
    def __init__(self, settings_dict, write_logger):
        # dataset settings
        #self.definition_file_dir = settings_dict['dataset_settings']['definition_file_dir']
        self.data_dir_lits = settings_dict['dataset_settings']['data_dir_lits']
        self.data_dir_spleen = settings_dict['dataset_settings']['data_dir_spleen']
        self.data_dir_pancreas = settings_dict['dataset_settings']['data_dir_pancreas']
        self.data_dir_kits = settings_dict['dataset_settings']['data_dir_kits']
        #self.data_dir_hippocampus = settings_dict['dataset_settings']['data_dir_hippocampus']


        self.load_masks = True
        # self.organ_to_seg = settings_dict['dataset_settings']['organ_to_seg']

        # pre processing settings
        self.pre_process = settings_dict['pre_processing_settings']['pre_process']
        self.apply_transformations = settings_dict['pre_processing_settings']['apply_transformations']
        self.augmentation=settings_dict['pre_processing_settings']['augmentation']
        if self.pre_process:
            self.min_clip_val = settings_dict['pre_processing_settings']['min_val']
            self.max_clip_val = settings_dict['pre_processing_settings']['max_val']
        else:
            self.min_clip_val = None
            self.max_clip_val = None

        # compilation settings
        self.optimizer = settings_dict['compilation_settings']['optimizer']
        self.gamma_decay = settings_dict['compilation_settings']['gamma_decay']
        self.loss = settings_dict['compilation_settings']['loss']
        self.loss_weights = settings_dict['compilation_settings']['loss_weights']
        self.lr_decay_step_size = settings_dict['compilation_settings']['lr_decay_step_size']
        self.lr_decay_policy = settings_dict['compilation_settings']['lr_decay_policy']
        self.initial_learning_rate = settings_dict['compilation_settings']['initial_learning_rate']
        self.weights_init = settings_dict['compilation_settings']['weights_init']
        self.weight_decaay = settings_dict['compilation_settings']['weight_decay']
        if self.optimizer == 'adam':
            self.beta_1 = settings_dict['compilation_settings']['beta_1']
            self.beta_2 = settings_dict['compilation_settings']['beta_2']

        # output_settings
        self.simulation_folder = settings_dict['output_settings']['simulation_folder']
        self.checkpoint_dir = os.path.join(self.simulation_folder, 'checkpoint')
        self.snapshot_dir = os.path.join(self.simulation_folder, 'snapshot')
        self.validation_snapshot_dir = os.path.join(self.simulation_folder, 'validation_snapshot')
        self.weights_dir=os.path.join(self.simulation_folder, 'weights')
        self.test_dir=os.path.join(self.simulation_folder, 'test')
        self.dices_dir=os.path.join(self.simulation_folder, 'dices')

        if not os.path.exists(self.simulation_folder):
            os.mkdir(self.simulation_folder)
        if not os.path.exists(self.snapshot_dir):
            os.mkdir(self.snapshot_dir)
        for task in ['/spleen','/pancreas','/lits','/kits']:
            if not os.path.exists((self.snapshot_dir+task)):
                os.mkdir((self.snapshot_dir+task))
        if not os.path.exists(self.validation_snapshot_dir):
            os.mkdir(self.validation_snapshot_dir)
        for task in ['/spleen','/pancreas','/lits','/kits']:
            if not os.path.exists((self.validation_snapshot_dir+task)):
                os.mkdir((self.validation_snapshot_dir+task))
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        if not os.path.exists(self.weights_dir):
            os.mkdir(self.weights_dir)
        if not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)
        if not os.path.exists(self.dices_dir):
            os.mkdir(self.dices_dir)
        for task in ['/spleen','/pancreas','/lits','/kits']:
            if not os.path.exists((self.test_dir+task)):
                os.mkdir((self.test_dir+task))

        # architecture settings
        self.use_skip=settings_dict['architecture_settings']['use_skip']
        self.encoder_name = settings_dict['architecture_settings']['encoder_name']
        self.encoder_depth = settings_dict['architecture_settings']['encoder_depth']
        self.encoder_weights = settings_dict['architecture_settings']['encoder_weights']
        self.decoder_use_batchnorm = settings_dict['architecture_settings']['decoder_use_batchnorm']
        self.decoder_channels = settings_dict['architecture_settings']['decoder_channels']
        self.in_channels = settings_dict['architecture_settings']['in_channels']
        self.classes = settings_dict['architecture_settings']['classes']
        self.dimension = settings_dict['architecture_settings']['dimension']
        self.input_size = settings_dict['architecture_settings']['input_size']
        if self.loss == 'CrossEntropyLoss':
            self.activation = 'identity'
        else:
            self.activation = settings_dict['architecture_settings']['activation']

        # training settings
        self.train_model = settings_dict['training_settings']['train_model']
        self.batch_size = settings_dict['training_settings']['batch_size']
        self.num_epochs = settings_dict['training_settings']['num_epochs']


    # copy experiment code
    def copy_code(self):
        formatted_time = time.strftime('%X %x %Z').replace(' ', '__').replace(':', '_').replace('/', '_')
        folder_name = 'code_snapshot_' + formatted_time
        target_path = os.path.join(self.simulation_folder, folder_name)
        self.logger.info('copying code snapshot to simulation folder : %s' % target_path)
        if not os.path.exists(target_path):
            self.logger.info('making dir  %s' % target_path)
            os.makedirs(target_path)
        for file in os.listdir():
            if file.endswith(".py"):
                file_path = (os.path.join(".", file))
                self.logger.info('copying file %s snapshot to simulation folder' % file_path)
                copy(file_path, target_path)

