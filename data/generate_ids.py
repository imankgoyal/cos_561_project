""" Script used for randomly generating the train, validation
    and test ids with their output
"""

import numpy as np

all_id = np.zeros((30, 2))
id_done = 0

while True:
    if id_done >= 30:
        break
    
    new_id = np.random.randint(12, size=(2))
    repeated = False
    for j in range(id_done):
        if np.all(all_id[j] == new_id):
            repeated = True
    
    if not repeated:
        all_id[id_done] = new_id
        id_done += 1

        
tr_id = all_id[0:20]
va_id = all_id[20:25]
te_id = all_id[25:30]

print(tr_id.transpose())
print(va_id.transpose())
print(te_id.transpose())

"""
Output:
[[10.  9.  2. 10.  4.  1.  3.  0.  2.  0.  3.  7.  1.  4.  4.  5.  6.  6.
   2.  0.]
 [ 7.  6.  5. 11.  9.  2.  4.  3.  0.  1.  8.  8.  7.  7.  3.  2.  7. 10.
   6.  5.]]
[[ 8. 11.  4.  9.  0.]
 [ 6.  5.  8.  8.  0.]]
[[ 1.  6. 10.  7.  5.]
 [ 8.  0.  8. 11. 10.]]
""""