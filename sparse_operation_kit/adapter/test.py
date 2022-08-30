import tensorflow as tf
import sparse_operation_kit.experiment as sok
import horovod.tensorflow as hvd

hvd.init()
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")
sok.init()

# var = tf.get_embedding_variable("var_0",
#                                 embedding_dim=3,
#                                 initializer=tf.ones_initializer(tf.float32),
#                                 partitioner=tf.fixed_size_partitioner(num_shards=4))

var = tf.get_embedding_variable("var_0",
                                embedding_dim=3,
                                initializer=tf.ones_initializer(tf.float32))
      
var.target_gpu = 1
print(var.shape)
print(type(var))
# emb = tf.nn.embedding_lookup(var, tf.cast([0,1,2,5,6,7], tf.int64))
emb = sok.lookup_sparse_for_deeprec(var, [[0,1,2]], hotness=[3, 4], combiners=['sum'])

fun = tf.multiply(emb, 2.0, name='multiply')
loss = tf.reduce_sum(fun, name='reduce_sum')
opt = tf.train.AdagradOptimizer(0.1)

g_v = opt.compute_gradients(loss)
train_op = opt.apply_gradients(g_v)

init = tf.global_variables_initializer()

sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
with tf.Session(config=sess_config) as sess:
  sess.run([init])
  print(sess.run([emb, train_op, loss]))
  print(sess.run([emb, train_op, loss]))
  print(sess.run([emb, train_op, loss]))