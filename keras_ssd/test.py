from keras_ssd.nets.ssd import SSD300

NUM_CLASSES = 21
input_shape = (300, 300, 3)
model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.summary()