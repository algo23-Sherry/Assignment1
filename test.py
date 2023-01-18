# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pymysql
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from threading import Thread
import threading
# import tensorflow as tf
# from tensorflow import keras
# from keras import regularizers
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.initializers import glorot_normal
# from tensorflow.keras.callbacks import *
import os,copy
import time,gc
from tqdm import tqdm
import datetime


import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.callbacks import *

