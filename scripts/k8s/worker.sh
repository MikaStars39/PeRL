# 定义 Pod 列表
PODS=("verl-30b-6node-gspo-1-99h7v" "verl-30b-6node-gspo-2-9szck" "verl-30b-6node-gspo-3-rz97c")

for POD in "${PODS[@]}"; do
  echo "正在处理 $POD ..."
  
  # 关键修改：去掉 -it，改用 -i (如果需要传流) 或者干脆什么都不带
  # 使用 nohup 确保即使你退出当前终端，Pod 内的任务也会继续
  kubectl exec $POD -n explore-train -- bash -c "cd /root/slime && source /etc/profile && sh /mnt/llm-train/users/explore-train/qingyu/PeRL/scripts/verl/k8s/worker.sh" > log_${POD}.out 2>&1 &
done

echo "所有任务已在后台发起，正在等待初始化..."
# 此时不需要 wait，因为 wait 会阻塞你的当前 shell 直到训练结束（可能很久）