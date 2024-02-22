import torch
import numpy as np
import exercise_blanks as exercise

data_manager = exercise.DataManager(batch_size=64)
s = data_manager.get_input_shape()
print(s[0])

x = np.zeros(5)
x[0] = 1

x1 = np.zeros(5)
x1[0] = 4

# x = torch.rand(5, 3)
# print(x)
# x1 = torch.reshape(x, (-1,))
# print(x1)
# x = torch.flatten(x)
# print(x)
