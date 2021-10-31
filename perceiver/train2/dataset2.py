

import enum
from typing import Any, Generator, Mapping, Optional, Sequence, Text, Tuple

import jax
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

from perceiver.train import autoaugment


Batch = Mapping[Text, np.ndarray]
MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)
AUTOTUNE = tf.data.experimental.AUTOTUNE

INPUT_DIM = 224  # The number of pixels in the image resize.


class Split(enum.Enum):
  """ImageNet dataset split."""
  TRAIN = 1
  TRAIN_AND_VALID = 2
  VALID = 3
  TEST = 4

  @classmethod
  def from_string(cls, name: Text) -> 'Split':
    return {'TRAIN': Split.TRAIN, 'TRAIN_AND_VALID': Split.TRAIN_AND_VALID,
            'VALID': Split.VALID, 'VALIDATION': Split.VALID,
            'TEST': Split.TEST}[name.upper()]

  @property
  def num_examples(self):
    return {Split.TRAIN_AND_VALID: 1281167, Split.TRAIN: 1271167,
            Split.VALID: 10000, Split.TEST: 50000}[self]


def _shard(
    split: Split, shard_index: int, num_shards: int) -> Tuple[int, int]:
  """Returns [start, end) for the given shard index."""
  assert shard_index < num_shards
  arange = np.arange(split.num_examples)
  shard_range = np.array_split(arange, num_shards)[shard_index]
  start, end = shard_range[0], (shard_range[-1] + 1)
  if split == Split.TRAIN:
    # Note that our TRAIN=TFDS_TRAIN[10000:] and VALID=TFDS_TRAIN[:10000].
    offset = Split.VALID.num_examples
    start += offset
    end += offset
  return start, end



def _to_tfds_split(split: Split) -> tfds.Split:
  """Returns the TFDS split appropriately sharded."""
  # NOTE: Imagenet did not release labels for the test split used in the
  # competition, so it has been typical at DeepMind to consider the VALID
  # split the TEST split and to reserve 10k images from TRAIN for VALID.
  if split in (
      Split.TRAIN, Split.TRAIN_AND_VALID, Split.VALID):
    return tfds.Split.TRAIN
  else:
    assert split == Split.TEST
    return tfds.Split.VALIDATION



def _random_generator():
  image = tf.random.uniform([INPUT_DIM, INPUT_DIM, 3], maxval=255, dtype=tf.int32)
  label = np.random.randint(10)
  yield 'filename_aaa', image, label


random_dataset = tf.data.Dataset.from_generator(
     _random_generator,
     output_signature=(
         tf.TensorSpec(shape=(), dtype=tf.int32, name = 'x'),
         tf.TensorSpec(shape=(150528,), dtype=tf.int32, name = 'Image'),
         tf.TensorSpec(shape=(), dtype=tf.int32))
    )



def load(
    split: Split,
    *,
    is_training: bool,
    # batch_dims should be:
    # [device_count, per_device_batch_size] or [total_batch_size]
    batch_dims: Sequence[int],
    augmentation_settings: Mapping[str, Any],
    # The shape to which images are resized.
    im_dim: int = INPUT_DIM,
    threadpool_size: int = 48,
    max_intra_op_parallelism: int = 1,
) -> Generator[Batch, None, None]:
  """Loads the given split of the dataset."""
  start, end = _shard(split, jax.host_id(), jax.host_count())

  im_size = (im_dim, im_dim)

  total_batch_size = np.prod(batch_dims)

  tfds_split = tfds.core.ReadInstruction(_to_tfds_split(split),
                                         from_=start, to=end, unit='abs')

  # ds = tfds.load('imagenet2012:5.*.*', split=tfds_split,
  #                decoders={'image': tfds.decode.SkipDecoding()})

  ds = random_dataset



  options = tf.data.Options()
  options.experimental_threading.private_threadpool_size = threadpool_size
  options.experimental_threading.max_intra_op_parallelism = (
      max_intra_op_parallelism)
  options.experimental_optimization.map_parallelization = True
  if is_training:
    options.experimental_deterministic = False
  ds = ds.with_options(options)

  if is_training:
    if jax.host_count() > 1:
      # Only cache if we are reading a subset of the dataset.
      ds = ds.cache()
    ds = ds.repeat()
    ds = ds.shuffle(buffer_size=10 * total_batch_size, seed=0)

  else:
    if split.num_examples % total_batch_size != 0:
      raise ValueError(f'Test/valid must be divisible by {total_batch_size}')


  # Mixup/cutmix by temporarily batching (using the per-device batch size):
  use_cutmix = augmentation_settings['cutmix']
  use_mixup = augmentation_settings['mixup_alpha'] is not None
  if is_training and (use_cutmix or use_mixup):
    inner_batch_size = batch_dims[-1]
    # Apply mixup, cutmix, or mixup + cutmix on batched data.
    # We use data from 2 batches to produce 1 mixed batch.
    ds = ds.batch(inner_batch_size * 2)
    # if not use_cutmix and use_mixup:
    #   ds = ds.map(my_mixup, num_parallel_calls=AUTOTUNE)
    # elif use_cutmix and not use_mixup:
    #   ds = ds.map(my_cutmix, num_parallel_calls=AUTOTUNE)
    # elif use_cutmix and use_mixup:
    #   ds = ds.map(my_mixup_cutmix, num_parallel_calls=AUTOTUNE)

    # Unbatch for further processing.
    ds = ds.unbatch()

  for batch_size in reversed(batch_dims):
    ds = ds.batch(batch_size)

  ds = ds.prefetch(AUTOTUNE)

  yield from tfds.as_numpy(ds)

