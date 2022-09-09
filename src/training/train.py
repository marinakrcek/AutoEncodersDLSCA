from src.training.data_augmentation import generate_data_augmentation


def train_model(model, dataset, epochs, batch_size, data_augmentation=False, model_type='',
                steps_per_epoch=0, n_batches_prof=0, n_batches_augmented=0, desync=False,
                gaussian_noise=False, time_warping=False):
    if data_augmentation:
        da_method = generate_data_augmentation(dataset.x_profiling, dataset.y_profiling, batch_size, model_type, n_batches_prof,
                                               n_batches_augmented, desync=desync, gaussian_noise=gaussian_noise, time_warping=time_warping)
        history = model.fit_generator(
            generator=da_method,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=2,
            validation_data=(dataset.x_attack, dataset.y_attack),
            validation_steps=1,
            callbacks=[])
    else:
        history = model.fit(
            x=dataset.x_profiling,
            y=dataset.y_profiling,
            batch_size=batch_size,
            verbose=2,
            epochs=epochs,
            shuffle=True,
            validation_data=(dataset.x_attack, dataset.y_attack),
            callbacks=[])

    return model, history


def train_ae_model(model, dataset, epochs, batch_size):
    # using only x (not y labels) for training
    # does not use attack data for validation
    valid_set_size = len(dataset.y_attack)
    history = model.fit(
            x=dataset.x_profiling[valid_set_size:],
            y=dataset.x_profiling[valid_set_size:],
            batch_size=batch_size,
            verbose=2,
            epochs=epochs,
            shuffle=True,
            validation_data=(dataset.x_profiling[:valid_set_size], dataset.x_profiling[:valid_set_size]),
            callbacks=[])
    return model, history


def train_ds_diff(model, x_train, y_train, epochs, batch_size, valid_set_size):
    history = model.fit(
        x=x_train[valid_set_size:],
        y=y_train[valid_set_size:],
        batch_size=batch_size,
        verbose=2,
        epochs=epochs,
        shuffle=True,
        validation_data=(x_train[:valid_set_size], y_train[:valid_set_size]),
        callbacks=[])
    return model, history
