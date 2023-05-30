def get_accuracy(x,y,num):
    
    counter = 0
    n_iter = 0
    for i in range(num) :
        if x[i] == y[i] : counter += 1
        n_iter += 1
    
    return counter/n_iter
