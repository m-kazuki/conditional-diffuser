pip install -f https://download.pytorch.org/whl/torch_stable.html \
                typed-argument-parser \
                scikit-image==0.17.2 \
                scikit-video==1.1.11 \
                gitpython \
                einops \
                pillow \
                free-mujoco-py \
                gym==0.18.0 \
                git+https://github.com/JannerM/d4rl.git@0e84a4d29db3ae501043215ce1d91843929f1949 \
                git+https://github.com/aravindr93/mjrl

pip install -e .
pip install tqdm
pip install gym==0.18.0