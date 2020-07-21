from tensorboardX import SummaryWriter
import numpy as np

writer = SummaryWriter("123123")

for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random())
    writer.add_scalar('Loss/test', np.random.random())
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)