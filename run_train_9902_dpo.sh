export NCCL_DEBUG=info export NCCL_SOCKET_IFNAME=bond4 export NCCL_IB_DISABLE=1
export NCCL_BLOCKING_WAIT=1
export NCCL_IB_TC=128
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=22

torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 --master_addr=10.11.11.19 --master_port=1108 train.py --train_args_file ./train_args/dpo/qlora/qwen1.5-14b-chat-dpo-qlora-xtop.json
