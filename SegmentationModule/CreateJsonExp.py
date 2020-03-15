import json
import os

experiments_dict = {'lr': [0.001, 0.0001],
                    'loss': ['BCE', 'CE'],
                    'task': ['8tasks'],
                    'encoder': ['densenet121', 'efficientnet-b7']}

def create_json(father_folder_path, exp_start_ind=0):
    exp_ind = exp_start_ind
    """
    a for loops over different experiments options should be added here
    """
    for task in experiments_dict['task']:
        for encoder_name in experiments_dict['encoder']:
            for loss in experiments_dict['loss']:
                for lr in experiments_dict['lr']:

                    exp_ind += 1
                    exp_dir = os.path.join(father_folder_path, 'exp_{}'.format(exp_ind))
                    if not os.path.exists(exp_dir):
                        os.mkdir(exp_dir)
                    json_data = {}

                    # dataset settings
                    json_data['dataset_settings'] = {'definition_file_dir': r'E:/Deep learning/Datasets_organized/sample_data',
                                                     'data_dir_lits': r'E:/Deep learning/Datasets_organized/sample_data/Lits',
                                                     'data_dir_prostate': r'E:/Deep learning/Datasets_organized/sample_data/Prostate',
                                                     # 'data_dir_brain': r'E:/Deep learning/Datasets_organized/Prepared_Data/BRATS',
                                                     # 'data_dir_hepatic_vessel': r'E:/Deep learning/Datasets_organized/Prepared_Data/Hepatic Vesel ',
                                                     # 'data_dir_spleen': r'E:/Deep learning/Datasets_organized/Prepared_Data/Spleen',
                                                     # 'data_dir_pancreas': r'E:/Deep learning/Datasets_organized/Prepared_Data/Pancreas',
                                                     # 'data_dir_left_atrial': r'E:/Deep learning/Datasets_organized/Prepared_Data/Left Atrial',
                                                     # 'data_dir_hippocampus': r'E:/Deep learning/Datasets_organized/Prepared_Data/Hippocampus',
                                                     # 'mask_labels_numeric': {'background': 0,  'liver': 1},
                                                     'task': task}

                    # pre processing settings
                    json_data['pre_processing_settings'] = {'clip': True,
                                                            'min_val': -300,
                                                            'max_val': 600,
                                                            'apply_transformations': False}

                    # compilation settings
                    json_data['compilation_settings'] = {'loss': loss,
                                                         'loss_weights': {'background': 1, 'liver': 10},
                                                         'weights_init': 'kaiming',
                                                         'initial_learning_rate': lr,
                                                         'gamma_decay': 0.5,
                                                         'lr_decay_policy': 'step',
                                                         'lr_decay_step_size': 5000,
                                                         'optimizer': 'adam',
                                                         'weight_decay': 0.0001,
                                                         'beta_1': 0.5,
                                                         'beta_2': 0.999}

                    # output_settings
                    json_data['output_settings'] = {
                        'simulation_folder': r'C:\Users\user_2018\Desktop\mip2\nimrod\DADE\seg_experiments\exp_{}'.format(exp_ind)}

                    # architecture settings
                    json_data['architecture_settings'] = {'encoder_name': 'efficientnet-b7',
                                                          'encoder_depth': 5,
                                                          'encoder_weights': 'imagenet',
                                                          'decoder_use_batchnorm': True,
                                                          'decoder_channels': [256, 128, 64, 32, 16],
                                                          'in_channels': 3,
                                                          'classes': 1,
                                                          'dimension': 2,
                                                          'activation': 'sigmoid',
                                                          'input_size': (3, 384, 384),
                                                          'use_skip': True}

                    # training settings
                    json_data['training_settings'] = {'train_model': True,
                                                      'batch_size': 4,
                                                      'num_epochs': 100}

                    # logger_settings
                    json_data['logger_settings'] = {'image_display_iter': 100,
                                                    'save_image_iter': 100,
                                                    'display_size': 4,
                                                    'log_iter': 100,
                                                    'epochs': 5,
                                                    'snapshot_save_iter': 10,
                                                    'save_loss_to_log': 2}


                    if encoder_name == 'efficientnet-b7':
                        json_data['training_settings']['batch_size'] = 2
                        json_data['logger_settings']['epochs'] = 5



                    file_path = os.path.join(exp_dir, 'exp_{}.json'.format(exp_ind))
                    with open(file_path, 'w') as f:
                        json.dump(json_data, f, indent=4)

if __name__== '__main__':
    folder_path = 'E:/Deep learning/Datasets_organized/Prepared_Data'
    exp_satart_ind = 0
    create_json(folder_path, exp_satart_ind)

# create_json('E:/Deep learning/Datasets_organized/Prepared_Data',1)
