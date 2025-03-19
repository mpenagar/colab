import tensorflow as tf
import subprocess

def detect_tpu():
  try:
    return tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
  except ValueError:
    return None

def detect_gpu():
  try:
      out = subprocess.check_output('nvidia-smi --query-gpu=name --format=csv,noheader'.split())
      return out.decode('utf-8').strip().split('\n')
  except Exception: # this command not being found can raise quite a few different errors depending on the configuration
      return []
 
def get_strategy():
  tpu = detect_tpu()
  if tpu :
    print(f'Running on a TPU w/{tpu.num_accelerators()["TPU"]} cores')
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    return tf.distribute.TPUStrategy(tpu)
  else :
    print('Not connected to a TPU runtime')
    gpu = detect_gpu()
    if gpu:
      print('Running on a GPU:',*gpu)
    else :
      print('Not connected to a GPU runtime')
    return tf.distribute.get_strategy()

def scope():
	return get_strategy().scope()