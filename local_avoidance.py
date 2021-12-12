import numpy as np
from path_following import motors

def is_obstacle(node):
    obstThr = 0     # obstacle threshold 
    
    # acquisition from the proximity sensors to detect obstacles
    obst = list(node["prox.horizontal"])
    
    return any([i > obstThr for i in obst])

def avoid_obstacle(speed_target, node):
    #error = 20    # décommenter pour que le thymio recule en tournant un petit peu quand il est juste en face
    
    # Create weight matrix of ANN :
    W = np.array([[2/3, 1/3, -3/2, -1/3, -2/3, 1, -1],[-2/3, -1/3, -3/2, 1/3, 2/3, -1, 1]])
    
    x = list(node["prox.horizontal"])
    
    # décommenter pour que le thymio recule en tournant un petit peu quand il est juste en face
    
    #if (all(sensor == 0 for sensor in (x[:2] + x[3:5]))):
        # Set desired motor speed with error
        #y = np.matmul(W, x)*3*initial_speed/5000 + [initial_speed+error, initial_speed-error]
        
    #else:
        # Set desired motor speed
    y = np.matmul(W, x)*2*speed_target/5000 + [speed_target, speed_target]
    
    node.send_set_variables(motors(int(y[0]), int(y[1])))

