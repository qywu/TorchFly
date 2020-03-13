from torchfly_dev.training.checkpointer.advanced_checkpointer import AdavancedCheckpointer 
import time 
import os 


# class Net(nn.Module):
# 	def __init__(self):
# 	 super().__init__()
# 	 self.model = nn.Sequential( nn.Linear(5000,4000) )
# 	def forward(self):
# 	 return 0




net = {"weights": 0, "bias":0}

saver = AdavancedCheckpointer(num_checkpoints_to_keep=2, keep_checkpoint_every_num_seconds=5)


saver.save_checkpoint( stamp='0', state = net.state_dict() )
saver.save_checkpoint( stamp='1', state = net.state_dict() )
# at this time there should be two models (0 and 1) saved 
saved_checkpoints = os.listdir('Checkpoints')
saved_checkpoints.sort()
assert (saved_checkpoints[0] == '0_state.pth')
assert (saved_checkpoints[1] == '1_state.pth')


saver.save_checkpoint( stamp='2', state = net.state_dict() )
# at this time there should be two models (1 and 2) saved, since less than 5 seconds thus 0 is deleted  
saved_checkpoints = os.listdir('Checkpoints')
saved_checkpoints.sort()
assert (saved_checkpoints[0] == '1_state.pth')
assert (saved_checkpoints[1] == '2_state.pth')


time.sleep(6)
saver.save_checkpoint( stamp='3', state = net.state_dict() )
saver.save_checkpoint( stamp='4', state = net.state_dict() )
# at this time there should be four models (1, 2  3, 4) saved, and 1,2 should be already in safe list
saved_checkpoints = os.listdir('Checkpoints')
saved_checkpoints.sort()
assert (saved_checkpoints[0] == '1_state.pth')
assert (saved_checkpoints[1] == '2_state.pth')
assert (saved_checkpoints[2] == '3_state.pth')
assert (saved_checkpoints[3] == '4_state.pth')


saver.save_checkpoint( stamp='5', state = net.state_dict() )
# at this time there should be four models (1, 2, 4,5) saved, and 1,2 should be already in safe list
saved_checkpoints = os.listdir('Checkpoints')
saved_checkpoints.sort()
assert (saved_checkpoints[0] == '1_state.pth')
assert (saved_checkpoints[1] == '2_state.pth')
assert (saved_checkpoints[2] == '4_state.pth')
assert (saved_checkpoints[3] == '5_state.pth')

print('test pass !')