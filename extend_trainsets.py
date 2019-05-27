import numpy as np
import os


def transfer(src_dir, ext_dir):
    files = os.listdir(src_dir)
    for f in files:
        ts = []
        path = os.path.join(src_dir, f)
        src = np.loadtxt(path)
        if src.shape != (100, 100):
            continue
        
        ts.append(src)
        
        ts.append(np.rot90(src, 1))     # rorate 90 <left> anti-clockwise
        ts.append(np.rot90(src, -1))    # rorate 90 <right> clockwise
        ts.append(np.rot90(src, 2))     # rorate 180 <left> anti-clockwise
        
        ver_flip = np.flipud(src)
        ts.append(ver_flip)
        
        ts.append(np.rot90(ver_flip, 1))
        ts.append(np.rot90(ver_flip, -1))
        ts.append(np.rot90(ver_flip, 2))
        
        for i in range(len(ts)):
            save_name = f.split(".")[0] + str("-") + str(i) + str(".txt")
            np.savetxt(os.path.join(ext_dir, save_name), ts[i])
        
        

if __name__ == "__main__":  
    good_examples_dir = "./data/good_data"
    bad_examples_dir = "./data/bad_data"
    ext_good_dir = "./data/good_data_extend"
    ext_bad_dir = "./data/bad_data_extend"    
    
    transfer(good_examples_dir, ext_good_dir)
    transfer(bad_examples_dir, ext_bad_dir)