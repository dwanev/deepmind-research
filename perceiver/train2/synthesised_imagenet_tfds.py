# coding=utf-8
# Copyright 2021 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Imagenet datasets."""

import io
import os
import tarfile
import glob
from PIL import Image
import numpy as np

from absl import logging

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """\
 hack to fake the imgenet dataset
"""

# Web-site is asking to cite paper from 2015.
# http://www.image-net.org/challenges/LSVRC/2012/index#cite
_CITATION = """\
none
"""

# file containing unique labels, 1 per line
_LABELS_FNAME = 'synimagenet/labels.txt'

# This file contains the validation labels, in the alphabetic order of
# corresponding image names (and not in the order they have been added to the
# tar file).
_VALIDATION_LABELS_FNAME = 'synimagenet/labels_validation.txt'

# From https://github.com/cytsai/ilsvrc-cmyk-image-list
CMYK_IMAGES = []
PNG_IMAGES = []

class SynImagenet(tfds.core.GeneratorBasedBuilder):
  """ fake imagenet like dataset"""

  VERSION = tfds.core.Version('0.0.1')
  SUPPORTED_VERSIONS = [
      tfds.core.Version('0.0.1'),
  ]
  RELEASE_NOTES = {
      '0.0.1':
          'hmmm....',
  }

  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  badluck
  """

  def _info(self):
    names_file = tfds.core.tfds_path(_LABELS_FNAME)
    print('looking for labels file in ', names_file)
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(encoding_format='jpeg'),
            'label': tfds.features.ClassLabel(names_file=names_file),
            'file_name': tfds.features.Text(),  # Eg: 'n15075141_54.JPEG'
        }),
        supervised_keys=('image', 'label'),
        homepage='http://google.com/',
        citation=_CITATION,
    )

  def _get_validation_labels(self, val_path):
    """Returns labels for validation.

    Args:
      val_path: path to TAR file containing validation images. It is used to
        retrieve the name of pictures and associate them to labels.

    Returns:
      dict, mapping from image name (excluding path and extension) (str) to label (class index number)(str).
    """

    labels_path = os.path.join( self.manual_dir, _VALIDATION_LABELS_FNAME)
    with tf.io.gfile.GFile(os.fspath(labels_path)) as labels_f:
        labels = labels_f.read().strip().splitlines()

    t_dir = os.path.join(self.manual_dir, 'synimagenet', 'validation', '*.jpg')
    file_list = glob.glob(t_dir)
    file_list2 = []
    for fn in file_list:
        fn2 = fn[fn.rfind('/')+1:-4]
        file_list2.append(fn2)

    file_list2.sort()

    return dict(zip(file_list2, labels))

  def _split_generators(self, dl_manager):
    self.manual_dir = dl_manager.manual_dir
    train_path = os.path.join(dl_manager.manual_dir, 'synimagenet/synimagenet_train.tar')
    val_path = os.path.join(dl_manager.manual_dir, 'synimagenet/synimagenet_valid.tar')
    test_path = None #os.path.join(dl_manager.manual_dir, 'synimagenet/ILSVRC2012_img_test.tar')
    splits = []
    print('train_path', train_path)
    _add_split_if_exists(
        split_list=splits,
        split=tfds.Split.TRAIN,
        split_path=train_path,
        dl_manager=dl_manager,
    )
    _add_split_if_exists(
        split_list=splits,
        split=tfds.Split.VALIDATION,
        split_path=val_path,
        dl_manager=dl_manager,
        validation_labels=self._get_validation_labels(val_path),
    )
    # _add_split_if_exists(
    #     split_list=splits,
    #     split=tfds.Split.TEST,
    #     split_path=test_path,
    #     dl_manager=dl_manager,
    #     labels_exist=False,
    # )
    if not splits:
      raise AssertionError(
          'ImageNet requires manual download of the data. Please download '
          'the data and place them into:\n'
          f' * train: {train_path}\n'
          f' * test: {test_path}\n'
          f' * validation: {val_path}\n'
          'At least one of the split should be available.')
    return splits

  def _fix_image(self, image_fname, image):
    """Fix image color system and format starting from v 3.0.0."""
    if self.version < '3.0.0':
      return image
    if image_fname in CMYK_IMAGES:
      image = io.BytesIO(
          tfds.core.utils.jpeg_cmyk_to_rgb(image.read()).tobytes())
    elif image_fname in PNG_IMAGES:
      image = io.BytesIO(tfds.core.utils.png_to_jpeg(image.read()).tobytes())
    return image

  def _load_image(self, infilename):
      img = Image.open(infilename)
      img.load()
      data = np.asarray(img, dtype="uint8")
      return data

  def _generate_examples(self,
                         archive,
                         validation_labels=None,
                         labels_exist=True):
    """Yields examples."""
    if not labels_exist:  # Test split
      for key, example in self._generate_examples_test(archive):
        yield key, example
    if validation_labels:  # Validation split
      for key, example in self._generate_examples_validation(
          archive, validation_labels):
        yield key, example
    # Training split. Main archive contains archives names after a synset noun.
    # Each sub-archive contains pictures associated to that synset.
    for fname, fobj in archive:
      # training split
      t_dir = self.data_dir
      t_dir = os.path.join(t_dir[:(t_dir.find('tensorflow_datasets')+20)], 'downloads/manual/synimagenet','train')
      # assuming fname is a directory
      fname = os.path.join(t_dir, '*.jpg')
      file_list = glob.glob(fname)
      file_list.sort()

      for image_fname in file_list:
          image = self._load_image(image_fname)
          fname = image_fname[image_fname.rfind('/')+1:]
          label = fname[:-4]
          label = 'CAT'
          record = {
              'file_name': image_fname,
              'image': image,
              'label': label,
          }
          yield image_fname, record


  def _generate_examples_validation(self, archive, labels):
    for fname, fobj in archive:
      record = {
          'file_name': fname,
          'image': fobj,
          'label': labels[fname],
      }
      yield fname, record

  def _generate_examples_test(self, archive):
    for fname, fobj in archive:
      record = {
          'file_name': fname,
          'image': fobj,
          'label': -1,
      }
      yield fname, record


def _add_split_if_exists(split_list, split, split_path, dl_manager, **kwargs):
  """Add split to given list of splits only if the file exists."""
  if not tf.io.gfile.exists(split_path):
    logging.warning(
        'ImageNet 2012 Challenge %s split not found at %s. '
        'Proceeding with data generation anyways but the split will be '
        'missing from the dataset...',
        str(split),
        split_path,
    )
  else:
    split_list.append(
        tfds.core.SplitGenerator(
            name=split,
            gen_kwargs={
                'archive': dl_manager.iter_archive(split_path),
                **kwargs
            },
        ),)





if __name__ == "__main__":

    # TODO get this running and expanding the tar files.

    print('running')

    sds = SynImagenet()
    print('running 2')

    sds.download_and_prepare()
    print('running 3')

    ds = sds.as_dataset()

    print('running 4')

    for r in ds:
        print(r)

    print('running 5')
