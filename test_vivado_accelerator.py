import os
import numpy as np
from hls4ml.converters import convert_from_keras_model
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
from callbacks import all_callbacks

import os

os.environ['PATH'] = '/tools/Xilinx/Vivado/2019.1/bin:' + os.environ['PATH']

# Load and scale dataset
data = fetch_openml('hls4ml_lhc_jets_hlf')
X, y = data['data'], data['target']
le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y, 5)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_val = scaler.fit_transform(X_train_val)
X_test = scaler.transform(X_test)
np.save('y_test.npy', y_test)
np.save('X_test.npy', X_test)
np.save('classes.npy', le.classes_, allow_pickle=True)

# Model
model = Sequential()
model.add(QDense(64, input_shape=(16,), name='fc1',
                 kernel_quantizer=quantized_bits(6,0,alpha=1), bias_quantizer=quantized_bits(6,0,alpha=1),
                 kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
model.add(QActivation(activation=quantized_relu(6), name='relu1'))
model.add(QDense(32, input_shape=(16,), name='fc2',
                 kernel_quantizer=quantized_bits(6,0,alpha=1), bias_quantizer=quantized_bits(6,0,alpha=1),
                 kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
model.add(QActivation(activation=quantized_relu(6), name='relu2'))
model.add(QDense(32, input_shape=(16,), name='fc3',
                 kernel_quantizer=quantized_bits(6,0,alpha=1), bias_quantizer=quantized_bits(6,0,alpha=1),
                 kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
model.add(QActivation(activation=quantized_relu(6), name='relu3'))
model.add(QDense(5, name='output',
                 kernel_quantizer=quantized_bits(6,0,alpha=1), bias_quantizer=quantized_bits(6,0,alpha=1),
                 kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
model.add(Activation(activation='softmax', name='softmax'))

# Training
from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule
from tensorflow_model_optimization.sparsity.keras import strip_pruning
pruning_params = {"pruning_schedule" : pruning_schedule.ConstantSparsity(0.75, begin_step=2000, frequency=100)}
model = prune.prune_low_magnitude(model, **pruning_params)

train = not os.path.exists('model/KERAS_check_best_model.h5')
if train:
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])
    callbacks = all_callbacks(stop_patience = 1000,
                              lr_factor = 0.5,
                              lr_patience = 10,
                              lr_epsilon = 0.000001,
                              lr_cooldown = 2,
                              lr_minimum = 0.0000001,
                              outputDir = 'training_dir')
    callbacks.callbacks.append(pruning_callbacks.UpdatePruningStep())
    model.fit(X_train_val, y_train_val, batch_size=1024,
              epochs=30, validation_split=0.25, shuffle=True,
              callbacks = callbacks.callbacks)
    # Save the model again but with the pruning 'stripped' to use the regular layer types
    model = strip_pruning(model)
    if not os.path.isdir('model'): 
        os.mkdir('model')
    model.save('model/KERAS_check_best_model.h5')
else:
    from tensorflow.keras.models import load_model
    from qkeras.utils import _add_supported_quantized_objects
    co = {}
    _add_supported_quantized_objects(co)
    model = load_model('model/KERAS_check_best_model.h5', custom_objects=co)

# Prediction
y_keras = model.predict(X.to_numpy())
np.save('y_qkeras.npy', y_keras)

# hls4ml
import hls4ml
hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'
hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name')

hls_config['Model'] = {}
hls_config['Model']['ReuseFactor'] = 64
hls_config['Model']['Strategy'] = 'Resource'
hls_config['Model']['Precision'] = 'ap_fixed<16,6>'
hls_config['LayerName']['fc1']['ReuseFactor'] = 64
hls_config['LayerName']['fc2']['ReuseFactor'] = 64
hls_config['LayerName']['fc3']['ReuseFactor'] = 64
input_data = os.path.join(os.getcwd(), 'X_test.npy')
output_predictions = os.path.join(os.getcwd(), 'y_qkeras.npy')

hls_model_axi_master = convert_from_keras_model(model=model, 
                                     backend='VivadoAccelerator',
                                     board='pynq-z1',
                                     io_type='io_stream',
                                     interface='axi_master',
                                     driver='c',
                                     input_data_tb='X_test.npy',
                                     output_data_tb='y_qkeras.npy',
                                     hls_config=hls_config, output_dir="test_backend_with_tb_axi_master")

hls_model_axi_master.build(csim=False, synth=True, export=True)

# Write inference data in a header file for baremetal application 
def write_header_file(X, y, y_keras, n_samples, filename="data.h"):
    header_file = open(filename, "w")
    (n_X_samples, n_X_inputs) = X.to_numpy().shape
    (n_y_samples, n_y_outputs) = y.shape
    (n_y_keras_samples, n_y_keras_outputs) = y_keras.shape

    header_file.write("#ifndef __DATA_H__\n")
    header_file.write("#define __DATA_H__\n")
    header_file.write("/* ouf of {} */\n".format(n_X_samples))
    header_file.write("#define N_SAMPLES {}\n".format(n_samples))
    header_file.write("\n")
    header_file.write("#define N_X_INPUTS {}\n".format(n_X_inputs))
    header_file.write("\n")
    header_file.write("const float src_data[N_SAMPLES*N_X_INPUTS] = {\n")
    for s in range(n_samples):
        header_file.write("    ")
        for i in range(n_X_inputs):
            header_file.write("{}, ".format(X.to_numpy()[s][i]))
        header_file.write("\n")
    header_file.write("};\n")
    header_file.write("\n")
    header_file.write("#define N_Y_OUTPUTS {}\n".format(n_y_outputs))
    header_file.write("\n")
    header_file.write("const float gld_data[N_SAMPLES*N_Y_OUTPUTS] = {\n")
    for s in range(n_samples):
        header_file.write("    ")
        for o in range(n_y_outputs):
            header_file.write("{}, ".format(y[s][o]))
        header_file.write("\n")
    header_file.write("};\n")
    header_file.write("\n")
    header_file.write("#define N_Y_KERAS_OUTPUTS {}\n".format(n_y_keras_outputs))
    header_file.write("")
    header_file.write("const float dst_data[N_SAMPLES*N_Y_KERAS_OUTPUTS] = {\n")
    for s in range(n_samples):
        header_file.write("    ")
        for o in range(n_y_keras_outputs):
            header_file.write("{}, ".format(y_keras[s][o]))
        header_file.write("\n")
    header_file.write("};\n")
    header_file.write("#endif\n")
    header_file.close()

write_header_file(X, y, y_keras, 8, "sdk/common/data.h")

hls4ml.report.read_vivado_report('test_backend_with_tb_axi_master/')

hls4ml.templates.VivadoAcceleratorBackend.make_bitfile(hls_model_axi_master)
