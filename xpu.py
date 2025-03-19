import tensorflow as tf

def detect_tpu():
  try:
    return tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
  except ValueError:
    return None

def detect_gpu():
  gpu = !nvidia-smi --query-gpu=name --format=csv,noheader 2> /dev/null
  return list(gpu)

def get_strategy():
  tpu = detect_tpu()
  if tpu :
    print(f'Running on a TPU w/{tpu.num_accelerators()["TPU"]} cores')
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    return tf.distribute.TPUStrategy(tpu)
  else :
    print('Not connected to a TPU runtime')
    gpu_info = !nvidia-smi --query-gpu=name --format=csv,noheader 2> /dev/null
    if gpu_info:
      print('Running on a GPU:',*gpu_info)
    else :
      print('Not connected to a GPU runtime')
    return tf.distribute.get_strategy()

def scope():
	return get_strategy().scope()