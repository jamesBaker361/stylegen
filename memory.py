import os
  
# Getting all memory using os.popen()
total_memory, used_memory, free_memory = map(
    int, os.popen('free -t -m').readlines()[-1].split()[1:])
  
# Memory usage
print("RAM memory % used:", round((used_memory/total_memory) ))
print('total memory {}'.format(total_memory/(1024**3)))