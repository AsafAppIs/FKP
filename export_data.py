# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from import_data import formating

def modelData_to_csv(trial):
  trial_line = []
  for timeframe in trial:
    trial_line += list(timeframe[0]) + list(timeframe[1]) + list(timeframe[2]) + list(timeframe[3])
  return np.array(trial_line)


def export_cnn_model_data(destination_path, X, Y):
  # format the samples in a 2D way
  X = formating(X, modelData_to_csv)
  # concatenate the samples and labels together
  data = np.concatenate((X, Y), axis=1)
  df = pd.DataFrame(data)

  df.to_csv(destination_path, header=False, index=False)
