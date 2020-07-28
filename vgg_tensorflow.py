from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Softmax, Conv2D, MaxPool2D, Dropout, BatchNormalization

model = Sequential([
        Conv2D(64, (3,3), padding = (1,1),activation = "prelu", input_data = (224,224,3), name = "conv1_1"),
        BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.001, name = "bn1_1"),
        Conv2D(64, (3,3), padding = (1,1), activation = "prelu", name = "conv1_2"),
        BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.001),
        MaxPool2D(kernel = (2,2), strides = (2,2), name = "pool1"),
        Dropout(0.25),
        
        Conv2D(128, (3,3), padding = (1,1),activation = "prelu", name = "conv2_1"),
        BatchNormalization(name = "bn2_1"),
        Conv2D(128, (3,3), padding = (1,1), activation = "prelu", name = "conv2_2"),
        BatchNormalization(name = "bn2_2"),
        MaxPool2D(pool_size = (2,2), strides = (2,2), name = "pool2"),
        Dropout(0.25),
        
        Conv2D(256, (3,3), padding = (1,1), activation = "prelu", name = "conv3_1"),
        BatchNormalization(name = "bn3_1"),
        Conv2D(256, (3,3), padding = (1,1), activation = "prelu", name= "conv3_2"),
        BatchNormalization(name = "bn3_2"),
        Conv2D(256, (3,3), padding = (1,1), activation = "prelu", name = "conv3_3"),
        BatchNormalization(name = "bn3_3"),
        MaxPool2D(pool_size = (2,2), strides = (2,2), name = "pool3"),
        Dropout(0.25),
        
        Conv2D(512, (3,3), padding = (1,1), activation = "prelu", name = "conv4_1"),
        BatchNormalization(name = "bn4_1"),
        Conv2D(512, (3,3), padding = (1,1), activation = "prelu", name = "conv4_2"),
        BatchNormalization(name = "bn4_2"),
        Conv2D(512, (3,3), padding = (1,1), activation = "prelu", name = "conv4_3"),
        BatchNormalization(name = "bn4_3"),
        MaxPool2D(pool_size = (2,2), strides = (2,2), name = "pool4"),
        Dropout(0.25),
        
        Conv2D(512, (3,3), padding = (1,1), activation = "prelu", name = "conv5_1"),
        BatchNormalization(name = "bn5_1"),
        Conv2D(512, (3,3), padding = (1,1), activation = "prelu", name = "conv5_2"),
        BatchNormalization(name = "bn5_2"),
        Conv2D(512, (3,3), padding = (1,1), activation = "prelu", name = "conv5_3"),
        BatchNormalization(name = "bn5_3"),
        MaxPool2D(pool_size = (2,2), strides = (2,2), name = "pool5"),
        Dropout(0.25),
        
        Flatten(name = "flatten"),
        
        Dense(4096, activation = "prelu", name = "fc1"),
        BatchNormalization(name = "bn6_1"),
        Dropout(0.5),
        
        Dense(4096, activation = "prelu", name = "fc2"),
        BatchNormalization(name = "bn7_1"),
        Dropout(0.5),
        
        Dense(units = 10, activation = "Softmax", name = "softmax")
        ])

model.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=False, name='SGD')
model.compile()
model.evaluate()
model.predict()

