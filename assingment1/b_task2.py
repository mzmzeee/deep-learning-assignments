from classes import *
from funxs import *
import numpy as np

SAMPLES = 500
LEARNING_RATE = 0.02
EPOCHS = 100
TRANING_PERCENT =0.4

X_train, X_test ,y_train, y_test = gen_xordata(SAMPLES , TRANING_PERCENT)