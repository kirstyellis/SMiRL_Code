Bootstrap: docker
From: gberseth/smirl:latest

%post
    mkdir -p /global/scratch
    mkdir -p /global/home/users/gberseth
    touch /bin/nvidia-smi
    touch /usr/bin/nvidia-smi
    touch /usr/bin/nvidia-debugdump
    touch /usr/bin/nvidia-persistenced
    touch /usr/bin/nvidia-cuda-mps-control
    touch /usr/bin/nvidia-cuda-mps-server
    mkdir /etc/dcv
    mkdir /var/lib/dcv-gl
    mkdir /usr/lib64
