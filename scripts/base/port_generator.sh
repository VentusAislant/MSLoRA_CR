# port_generator.sh

# 生成一个随机端口
generate_random_port() {
  local min_port=1024
  local max_port=65535
  while :; do
    port=$(shuf -i ${min_port}-${max_port} -n 1)
    # 检查端口是否被占用
    if ! ss -ltn | awk '{print $4}' | grep -q ":$port$"; then
      echo $port
      return
    fi
  done
}