import os
import nibabel as nib
import nibabel.processing
import numpy as np
import tensorflow as tf

def preprocess(path, config):

    scan = nib.load(path)
    aff  = scan.affine
    vol  = np.int16(scan.get_fdata())
    print(f'shape = {vol.shape}, path = {path}')

    # Resamplea volumen y affine a un nuevo shape
    new_zooms  = np.array(scan.header.get_zooms()) * config['new_z']
    new_shape  = np.array(vol.shape) // config['new_z']
    new_affine = nibabel.affines.rescale_affine(aff, vol.shape, new_zooms, new_shape)
    scan       = nibabel.processing.conform(scan, new_shape, new_zooms)
    ni_img     = nib.Nifti1Image(scan.get_fdata(), new_affine)
    vol        = np.int16(ni_img.get_fdata())

    return vol

def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    #volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

def get_ds(config):

    # Formar lista con las rutas de los pacientes de la base de datos BraTS
    fcd_subjects_path = []
    fcd_subjects = next(os.walk(config['reg_iatm_fcd']))[1]
    for fcd_subject in fcd_subjects:
        if 'FCD' in fcd_subject:
            studies = next(os.walk(os.path.join(config['reg_iatm_fcd'], fcd_subject)))[1]
            for study in studies:
                #if 'AIM' not in study:
                files = next(os.walk(os.path.join(config['reg_iatm_fcd'], fcd_subject, study)))[2]
                for file_ in files:
                    if 'nii.gz' in file_ and not('json'      in file_ or 
                                                 'Displasia' in file_ or 
                                                 'DISPLASIA' in file_ or
                                                 'mask'      in file_ or
                                                 'Eq_1'      in file_):
                        path = os.path.join(config['reg_iatm_fcd'],
                                                   fcd_subject,
                                                   study,
                                                   file_)
                        fcd_subjects_path.append(path)

    print(f'\nTotal mris con fcd = {len(fcd_subjects_path)}')

    # Lista con las rutas a las MRI de los controles
    control_paths = []
    ctrl_subjects = next(os.walk(config['iatm_ss_controls']))[1]
    for ctrl_subject in ctrl_subjects:
        # print(f'{ctrl_subject}')
        studies = next(os.walk(os.path.join(config['iatm_ss_controls'], ctrl_subject)))[1]
        for study in studies:
            # print(f'\t{ctrl_subject}')
            files = next(os.walk(os.path.join(config['iatm_ss_controls'], ctrl_subject, study, 'NIFTI')))[2]
            for file_ in files:
                # print(f'\t\t{file_}\n\n')
                p = os.path.join(config['iatm_ss_controls'], 
                                 ctrl_subject, study, 
                                 'NIFTI', 
                                 file_
                )
                control_paths.append(p)

    print(f'\nTotal sujetos de control = {len(control_paths)}\n')

    print(f'Estudios FCD incluidos en entrenamiento y validacion: \n')
    for s in fcd_subjects_path[:-config['n_test']]:
        print(f'{s}')
    print(f'\nEstudios FCD excluidos (para test): \n')
    for t in fcd_subjects_path[-config['n_test']:]:
        print(f'{t}')

    fcd_scans  = np.array([preprocess(path, config) for path in fcd_subjects_path[:config['n_heads']]])
    ctrl_scans = np.array([preprocess(path, config) for path in control_paths[:config['n_heads']]])

    # Vectores de etiquetas
    fcd_labels = np.array([1 for _ in range(len(fcd_scans))])
    ctrl_labels = np.array([0 for _ in range(len(ctrl_scans))])

    print(f'\nCantidad de volumenes FCD = {len(fcd_scans)}')
    print(f'Cantidad de sujetos de control = {len(ctrl_scans)}\n')

    # Dividir los datos en train validacion y test
    x_train = np.concatenate((fcd_scans[:config['n_train']], 
                              ctrl_scans[:config['n_train']]), 
                              axis=0)
    y_train = np.concatenate((fcd_labels[:config['n_train']], 
                              ctrl_labels[:config['n_train']]), 
                              axis=0)

    x_val = np.concatenate((fcd_scans[config['n_train']:config['n_train']+config['n_val']], 
                            ctrl_scans[config['n_train']:config['n_train']+config['n_val']]), 
                            axis=0)
    y_val = np.concatenate((fcd_labels[config['n_train']:config['n_train']+config['n_val']], 
                            ctrl_labels[config['n_train']:config['n_train']+config['n_val']]), 
                            axis=0)

    x_test = np.concatenate(( fcd_scans[config['n_train']+config['n_val']:], 
                             ctrl_scans[config['n_train']+config['n_val']:]), axis=0)
    y_test = np.concatenate(( fcd_labels[config['n_train']+config['n_val']:], 
                             ctrl_labels[config['n_train']+config['n_val']:]), axis=0)

    # print(f'xtrain shape = {x_train.shape}')
    # print(f'y_train shape = {y_train.shape}')
    # print(f'x_val shape = {x_val.shape}')
    # print(f'y_val shape = {y_val.shape}')
    # print(f'x_test shape = {x_test.shape}')
    # print(f'y_test shape = {y_test.shape}')

    # Define data loaders.
    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # Augment the on the fly during training.
    train_dataset = (
        train_loader.shuffle(len(x_train))
        .map(train_preprocessing)
        .batch(config['batch_size'])
        .prefetch(2)
    )
    # Only rescale.
    validation_dataset = (
        validation_loader.shuffle(len(x_val))
        .map(train_preprocessing)
        .batch(config['batch_size'])
        .prefetch(2)
    )

    test_dataset = (
        test_loader.shuffle(len(x_test))
        .map(train_preprocessing)
        .batch(config['batch_size'])
        .prefetch(2)
    )

    return train_dataset, validation_dataset, test_dataset



