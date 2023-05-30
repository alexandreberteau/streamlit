import numpy as np
import numpy.lib.recfunctions as rfn

def counter(data,range = -1):
    cleaned_data = data.dropna()
    catégories = cleaned_data.unique()
    décompte =  dict.fromkeys(catégories, 0)

    for secteur in décompte.keys():
        for ele in cleaned_data[:range] :
            if ele =="" : pass
            elif ele == secteur :
                décompte[ele] += 1
            else :
                pass

    
    print(décompte)
    return(décompte)

