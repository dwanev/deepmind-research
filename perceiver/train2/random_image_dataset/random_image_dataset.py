"""random_image_dataset dataset."""

import tensorflow as tf
import jax
import enum
from pathlib import Path
from typing import Any, Generator, Mapping, Optional, Sequence, Text, Tuple
import tensorflow_datasets as tfds
import numpy as np

# TODO(random_image_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(random_image_dataset): BibTeX citation
_CITATION = """
"""

INPUT_DIM = 224 # ?
AUTOTUNE = tf.data.experimental.AUTOTUNE


class RandomImageDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for random_image_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  Batch = Mapping[Text, np.ndarray]

  class Split(enum.Enum):
      """ImageNet dataset split."""
      TRAIN = 1
      TRAIN_AND_VALID = 2
      VALID = 3
      TEST = 4

      @classmethod
      def from_string(cls, name: Text) -> 'Split':
          return {'TRAIN': RandomImageDataset.Split.TRAIN, 'TRAIN_AND_VALID': RandomImageDataset.Split.TRAIN_AND_VALID,
                  'VALID': RandomImageDataset.Split.VALID, 'VALIDATION': RandomImageDataset.Split.VALID,
                  'TEST': RandomImageDataset.Split.TEST}[name.upper()]

      @property
      def num_examples(self):
          return {RandomImageDataset.Split.TRAIN_AND_VALID: 1281167, RandomImageDataset.Split.TRAIN: 1271167,
                  RandomImageDataset.Split.VALID: 10000, RandomImageDataset.Split.TEST: 50000}[self]

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(random_image_dataset): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(None, None, 3)),
            'label': tfds.features.ClassLabel(names=['no', 'yes']),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(random_image_dataset): Downloads the data and defines the splits

    # extracted_path = dl_manager.download_and_extract('https://todo-data-url')
    # dl_manager returns pathlib-like objects with `extracted_path.read_text()`,
    # `extracted_path.iterdir()`,...

    extracted_path = Path('/home/dwane/projects/tmp/hdd2_data/random_image')

    return {
        'train': self._generate_examples(path=extracted_path / 'train_images'),
        'test': self._generate_examples(path=extracted_path / 'test_images'),

    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(random_image_dataset): Yields (key, example) tuples from the dataset
    for f in path.glob('*.jpg'):
      yield str(f), {
          'image': f,
          'label': 'yes',
      }

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
      # start, end = RandomImageDataset._shard(split, jax.host_id(), jax.host_count())

      im_size = (im_dim, im_dim)

      total_batch_size = np.prod(batch_dims)

      # tfds_split = tfds.core.ReadInstruction(_to_tfds_split(split),
      #                                        from_=start, to=end, unit='abs')

      # ds = tfds.load('imagenet2012:5.*.*', split=tfds_split,
      #                decoders={'image': tfds.decode.SkipDecoding()})

      ds = tfds.load('random_image_dataset',
                     decoders={'image': tfds.decode.SkipDecoding()})


      # options = tf.data.Options()
      # options.experimental_threading.private_threadpool_size = threadpool_size
      # options.experimental_threading.max_intra_op_parallelism = (
      #     max_intra_op_parallelism)
      # options.experimental_optimization.map_parallelization = True
      # if is_training:
      #     options.experimental_deterministic = False
      # ds = ds.with_options(options)

      if is_training:
          if jax.process_count() > 1:
              # Only cache if we are reading a subset of the dataset.
              ds = ds.cache()
          # ds = ds.repeat()
          # ds = ds.shuffle(buffer_size=10 * total_batch_size, seed=0)

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

      # for batch_size in reversed(batch_dims):
      #     ds = ds.batch(batch_size)

      # ds = ds.prefetch(AUTOTUNE)

      yield from tfds.as_numpy(ds)

  def _shard(
          split: Split, shard_index: int, num_shards: int) -> Tuple[int, int]:
      """Returns [start, end) for the given shard index."""
      assert shard_index < num_shards
      arange = np.arange(split.num_examples)
      shard_range = np.array_split(arange, num_shards)[shard_index]
      start, end = shard_range[0], (shard_range[-1] + 1)
      if split == RandomImageDataset.Split.TRAIN:
          # Note that our TRAIN=TFDS_TRAIN[10000:] and VALID=TFDS_TRAIN[:10000].
          offset = RandomImageDataset.Split.VALID.num_examples
          start += offset
          end += offset
      return start, end


if __name__ == '__main__':
    print('hello')
    ds  = RandomImageDataset()
    ds.download_and_prepare()

    d = ds.as_dataset()
    print(len(d))

    print('loading....')
    per_device_batch_size = 2
    # split = RandomImageDataset.Split.TRAIN,
    ds_generator = ds.load(  is_training=True, batch_dims = [jax.local_device_count(), per_device_batch_size], augmentation_settings = {'cutmix':False, 'mixup_alpha':None} )

    print(ds_generator)

    for x in ds_generator:
        print(x)