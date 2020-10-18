import os

n = 4
for i in range(n):
    command_line = 'python 1main.py %s &'%i
    #command_line = 'python 1main_LR.py %s &'%i
    print(command_line)
    os.system(command_line)

