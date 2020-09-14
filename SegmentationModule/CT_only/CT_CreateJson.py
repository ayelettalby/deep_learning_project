import json
import os

lr = 0.00001
initial_weights = 'imagenet' #None
augmentation = True
encoder_name =  'densenet121'


def create_json(father_folder_path, exp_start_ind=0):
    exp_ind = exp_start_ind
    """
    a for loops over different experiments options should be added here
    """

    exp_dir = os.path.join(father_folder_path, 'CT_experiments/exp_{}'.format(exp_ind))
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    json_data = {}

    # dataset settings


    json_data['dataset_settings'] = {'definition_file_dir': father_folder_path,
                                   'data_dir_lits': father_folder_path +'\Lits',
                                    'data_dir_spleen': father_folder_path + '\Spleen',
                                     'data_dir_pancreas': father_folder_path + '\Pancreas',
                                     'data_dir_kits': father_folder_path + '\kits',
                                     }

    # pre processing settings
    json_data['pre_processing_settings'] = {'pre_process': True, #[True, False],
                                            'min_val': -300,
                                            'max_val': 600,
                                            'apply_transformations': False,
                                            'augmentation': augmentation
                                            }

    # compilation settings
    json_data['compilation_settings'] = {'loss': 'CrossEntropyLoss',
                                         'loss_weights': {'background': 1, 'organ': 10},
                                         'weights_init': initial_weights,
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
        'simulation_folder': father_folder_path + '\CT_experiments\exp_{}'.format(exp_ind)}

    # architecture settings
    json_data['architecture_settings'] = {'encoder_name': encoder_name,
                                          'encoder_depth': 5,
                                          'encoder_weights': initial_weights,
                                          'decoder_use_batchnorm': True,
                                          'decoder_channels': [256, 128, 64, 32, 16],
                                          'in_channels': 3,
                                          'classes': 2,
                                          'dimension': 2,
                                          'activation': 'softmax',
                                          'input_size': (3, 384, 384),
                                          'use_skip': None}

    # training settings
    json_data['training_settings'] = {'train_model': True,
                                      'batch_size': 1,
                                      'num_epochs': 100}

    # logger_settings
    json_data['logger_settings'] = {'image_display_iter': 100,
                                    'save_image_iter': 100,
                                    'display_size': 4,
                                    'log_iter': 100,
                                    'epochs': 5,
                                    'snapshot_save_iter': 10,
                                    'save_loss_to_log': 2}


    # if encoder_name == 'efficientnet-b7':
    #     json_data['training_settings']['batch_size'] = 2
    #     json_data['logger_settings']['epochs'] = 5



    file_path = os.path.join(exp_dir, 'exp_{}.json'.format(exp_ind))
    with open(file_path, 'w') as f:
        json.dump(json_data, f, indent=4)

if __name__== '__main__':
    user='remote' #ayelet
    if user=='ayelet':
        folder_path = r'C:\Users\Ayelet\Desktop\school\fourth_year\deep_learning_project\ayelet_shiri\sample_Data'
    elif user=='remote':
        folder_path= r'G:\Deep learning\Datasets_organized\Prepared_Data'
    elif user=='shiri':
        folder_path=r'F:/Prepared Data'
    exp_start_ind = 3
    create_json(folder_path, exp_start_ind)

# create_json('E:/Deep learning/Datasets_organized/Prepared_Data',1)
