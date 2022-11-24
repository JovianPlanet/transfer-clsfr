import os

def get_parameters(mode):

    # mode = 'reg' # available modes: 'train', 'test'

    pretrained_model = 'tumor_clf.h5'

    pretrained_model_path = os.path.join('./pretrained_model', pretrained_model)

    fcd_model = 'fcd_clf.h5'

    iatm_fcd = os.path.join('/media',
                            'davidjm',
                            'Disco_Compartido',
                            'david',
                            'datasets',
                            'IATM-Dataset'
    )

    iatm_controls = os.path.join('/media',
                                 'davidjm',
                                 'Disco_Compartido',
                                 'david',
                                 'datasets',
                                 'IATM-Dataset',
                                 'sujetos_proyecto_controles'
    )

    iatm_ss_controls = os.path.join('/home',
                                    'davidjm',
                                    'Downloads',
                                    'Reg-IATM',
                                    'sujetos_proyecto_controles'
    )

    model_dims = (128, 128, 64)
    lr = 0.0001
    epochs = 100
    batch_size = 4
    n_freeze = 2
    n_train = 18
    n_val = 4
    n_test = 4

    return {'mode'              : mode,
            'pretranined_model' : pretrained_model_path,
            'fcd_model'         : fcd_model,
            'iatm_fcd'          : iatm_fcd,
            'iatm_controls'     : iatm_controls,
            'iatm_ss_controls'  : iatm_ss_controls,
            'model_dims'        : model_dims,
            'lr'                : lr,
            'epochs'            : epochs,
            'batch_size'        : batch_size,
            'n_freeze'          : n_freeze,
            'n_train'           : n_train,
            'n_val'             : n_val,
            'n_test'            : n_test,
    }