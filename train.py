import os
from tensorflow import keras

from get_data import get_ds

def train(config):

    try:
        base_model = keras.models.load_model(config['pretranined_model'])
    except:
        print(f'Modelo no encontrado')
        return

    print(f'Model summary = {base_model.summary()}')

    base_model.trainable = False

    inputs = keras.Input(shape=config['model_dims'])

    outputs = keras.layers.Dense(units=1, 
                                 activation='sigmoid', 
                                 name='transfer-dense')(base_model.layers[-config['n_freeze']].output)

    model = keras.Model(base_model.input, outputs)

    print(f'\nTransfer learning model summary = \n\n{model.summary()}')

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=[keras.metrics.BinaryAccuracy()])

    # Make datasets
    train_ds, val_ds = get_ds(config)[:2]

    # Define callbacks.
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        config['fcd_model'], save_best_only=True
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

    model.fit(train_ds, 
              epochs=config['epochs'], 
              callbacks=[checkpoint_cb, early_stopping_cb], 
              validation_data=val_ds
    )








