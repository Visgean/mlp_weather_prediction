from weatherbench.train_nn import build_cnn

filters = [64, 64, 64, 64, 1]
kernels = [5, 5, 5, 5, 5]

model = build_cnn(filters, kernels, input_shape=(32, 64, 11), activation='elu', dr=0)

# model.compile(keras.optimizers.Adam(lr), 'mse')
print(model.summary())
