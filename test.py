import os
import numpy as np
from tensorflow import keras
import nibabel as nib
import nibabel.processing
from get_data import get_ds

def test(config):

    try:
        model = keras.models.load_model(config['fcd_model'])
    except:
        print(f'Modelo no encontrado')
        return

    print(f'Model summary = {model.summary()}')

    test_ds = get_ds(config)[2]

    print(f'\nModel metrics: {model.metrics_names}\n')

    print(f'Evaluacion = {model.evaluate(test_ds)}\n')

    print(f'Predicciones = {model.predict(test_ds):.3f}')
    predictions = model.predict(test_ds)

    for pred in range(predictions.shape[0]):
        print(f'Predicciones = {predictions[pred][0]:.3f}')

    # control_paths = []
    # ctrl_subjects = next(os.walk(config['iatm_ss_controls']))[1]
    # for ctrl_subject in ctrl_subjects:
    #     print(f'{ctrl_subject}')
    #     studies = next(os.walk(os.path.join(config['iatm_ss_controls'], ctrl_subject)))[1]
    #     for study in studies:
    #         print(f'\t{ctrl_subject}')
    #         files = next(os.walk(os.path.join(config['iatm_ss_controls'], ctrl_subject, study, 'NIFTI')))[2]
    #         for file_ in files:
    #             print(f'\t\t{file_}\n\n')
    #             p = os.path.join(config['iatm_ss_controls'], ctrl_subject, study, 'NIFTI', file_)
    #             #try:
    #             control = nib.load(p)
    #             #except:
    #             #    continue
    #             control = nibabel.processing.conform(control, (128, 128, 64))
    #             control = control.get_fdata()
    #             control = np.expand_dims(control, 0)
    #             print(f'{model.predict(control)}')


    # print(f'\nPredicciones para el {ds} dataset:\n')








