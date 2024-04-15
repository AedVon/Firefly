export NCCL_DEBUG=info export NCCL_SOCKET_IFNAME=bond4 export NCCL_IB_DISABLE=1
export NCCL_BLOCKING_WAIT=1
export NCCL_IB_TC=128
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=22

torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 --master_addr=10.11.11.19 --master_port=1108 train.py --train_args_file ./train_args/sft/qlora/qwen1.5-14b-chat-sft-qlora-xtop.json
