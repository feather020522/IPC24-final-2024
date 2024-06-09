# IPC24-FINAL-2024

## Setup Environment (PyTorch)
Reference : 
https://gist.github.com/mhubii/1c1049fb5043b8be262259efac4b89d5
https://pytorch.org/cppdocs/installing.html

cuda version with ```nvcc --version ```is 11.7

1. make a folder with any name (outside of our workspace folder) i.e. create downloads folder
``` 
mkdir downloads
``` 
2. Move to the folder made and run the command below 
```python
cd downloads
wget https://repo.continuum.io/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
sh Anaconda3-2024.02-1-Linux-x86_64.sh
```
3. follow instruction to install conda, if you typed 'no' in the end in the shell
```
eval "$(/root/anaconda3/bin/conda shell.bash hook)" 
```

4. Create environment
```
conda create --name py39_torch python=3.9
conda activate py39_torch
```

5. Run command below to download pytorch
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```
6. go to project directory (assuming that CMakeLists.txt already exist)
```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="${HOME}/anaconda3/envs/py39_torch_old/lib/python3.9/site-packages/torch" ..
cmake --build . --config Release
```

7. run main in build folder
```
./main
```
