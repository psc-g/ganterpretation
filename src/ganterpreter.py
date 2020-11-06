import cStringIO
import IPython.display
import numpy as np
import PIL.Image
from scipy.io import wavfile
from scipy.stats import truncnorm
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio
from IPython.display import HTML, display
import cv2
import os

import categories

def truncated_z_sample(batch_size, dim_z, truncation=1., seed=None):
  state = None if seed is None else np.random.RandomState(seed)
  values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state)
  return truncation * values


def running_mean(x, N):
  """Compute running mean.
  """
  cumsum = np.cumsum(np.insert(x, 0, 0)) 
  return (cumsum[N:] - cumsum[:-N]) / float(N)


def get_inflection_points(arr, threshold, absolute_threshold=8e-2,
                          type='both', rolling_length=200):
  inflection_points = [0]
  i = 0
  while i < len(arr) - rolling_length - 1:
    prev_mean = np.mean(arr[i:i+rolling_length])
    curr_pos = i + rolling_length + 1
    next_mean = np.mean(arr[curr_pos+1:curr_pos+rolling_length+1])
    is_peak = (
            np.sign(arr[curr_pos] - prev_mean) ==
            np.sign(arr[curr_pos] - next_mean) and
            np.sign(arr[curr_pos] - arr[curr_pos-1]) ==
            np.sign(arr[curr_pos] - arr[curr_pos+1])) 
    if (is_peak and
        np.abs(arr[curr_pos] - prev_mean) > threshold and
        np.abs(arr[curr_pos] - next_mean) > threshold):
      if ((type == 'min' and (arr[curr_pos] > arr[curr_pos-1] or
          arr[curr_pos] > absolute_threshold)) or
          (type == 'max' and arr[curr_pos] < arr[curr_pos-1])):
        i += rolling_length
        continue
      inflection_points.append(curr_pos)
      i += rolling_length
    else:
      i += 1
  inflection_points.append(len(arr) - 1)
  return np.array(inflection_points)


def get_alphas(arr):
  cumsum = np.cumsum(arr)
  total_sum = np.sum(arr)
  return cumsum / total_sum

MODULE_PATHS = {
    'biggan-128': 'https://tfhub.dev/deepmind/biggan-deep-128/1',
    'biggan-256': 'https://tfhub.dev/deepmind/biggan-deep-256/1',
    'biggan-512': 'https://tfhub.dev/deepmind/biggan-deep-512/1'
}


class GANterpreter(object):
  """GANterpreter."""

  def __init__(self, model_type='biggan-512', selected_categories=[],
               verbose=False):
    """Create GANterpreter object.

    Args:
      model_type: str, specifies which of the BigGANs to use. Can be one of
        {'biggan-128', 'biggan-256', 'biggan-512'}
    """
    module_path = MODULE_PATHS[model_type]
    # Load the BigGAN model.
    tf.reset_default_graph()
    if verbose:
      print('Loading BigGAN module from: {}'.format(module_path))
    module = hub.Module(module_path)
    inputs = {k: tf.placeholder(v.dtype, v.get_shape().as_list(), k)
              for k, v in module.get_input_info_dict().iteritems()}
    self.output = module(inputs)
    # Initialize the session.
    initializer = tf.global_variables_initializer()
    self.sess = tf.Session()
    self.sess.run(initializer)
    # Necessary functions
    self.input_z = inputs['z']
    self.input_y = inputs['y']
    self.input_trunc = inputs['truncation']
    self.dim_z = self.input_z.shape.as_list()[1]
    self.vocab_size = self.input_y.shape.as_list()[1]
    self.framerate = None
    self.sounddata = None
    self.selected_categories = (
            [] if selected_categories is None else selected_categories)

  def load_wav_file(self, wav_path, verbose=False):
    """
    Loads wav file data.
    """
    self.framerate, self.sounddata = wavfile.read(wav_path)
    self.sounddata = self.sounddata.T
    if verbose:
      total_time = self.sounddata.shape[1]/self.framerate
      print('Sample rate: {}Hz'.format(self.framerate))
      print('Total time: {}s'.format(total_time))

  def compute_spectrogram(self, channel_to_use='Left',
                          inflection_threshold=1e-2, verbose=False):
    """Compute the spectrogram and inflection points from loaded audio."""
    assert self.framerate is not None
    assert self.sounddata is not None
    channel = 0 if channel_to_use == 'Left' else 1
    spectrum, _, _, _ = plt.specgram(self.sounddata[channel], Fs=self.framerate)
    self.tv_diffs = [np.mean(np.abs(spectrum[:, i] - spectrum[:, i + 1]))
            for i in range(spectrum.shape[1] - 1)]
    self.tv_diffs /= max(self.tv_diffs)
    self.running_mean_tv_diffs = running_mean(self.tv_diffs, 256)
    self.running_mean_tv_diffs = np.append(
            np.array([self.running_mean_tv_diffs[0]] * 256),
            self.running_mean_tv_diffs)
    self.inflection_points = get_inflection_points(
            self.running_mean_tv_diffs, inflection_threshold, type='min')
    self.alphas = np.array([])
    next_inflection_idx = 1
    i = 1
    while i < len(self.running_mean_tv_diffs):
      next_i = self.inflection_points[next_inflection_idx]
      self.alphas = np.concatenate(
              (self.alphas, get_alphas(self.running_mean_tv_diffs[i:next_i])))
      i = next_i
      next_inflection_idx += 1
      if next_inflection_idx >= len(self.inflection_points):
        break
    self.alphas = np.concatenate((self.alphas, [1.0]))
    if verbose:
      print('Number of inflection points: {}'.format(
          len(self.inflection_points)))

  def fill_selected_categories(self):
    for i in range(len(self.selected_categories), len(self.inflection_points)):
      self.selected_categories.append(np.random.randint(
          len(categories.ALL_CATEGORIES)))
    self.selected_categories = (
            self.selected_categories[:len(self.inflection_points)])

  def one_hot_if_needed(self, label):
    label = np.asarray(label)
    if len(label.shape) <= 1:
      label = self.one_hot(label)
    assert len(label.shape) == 2
    return label

  def get_new_target_category(self):
    category = categories.ALL_CATEGORIES[np.random.randint(
        len(categories.ALL_CATEGORIES))]
    return self.one_hot([int(category.split(')')[0])])

  def one_hot(self, index):
    index = np.asarray(index)
    if len(index.shape) == 0:
      index = np.asarray([index])
    assert len(index.shape) == 1
    num = index.shape[0]
    one_hot_category = np.zeros((num, self.vocab_size), dtype=np.float32)
    one_hot_category[np.arange(num), index] = 1
    return one_hot_category

  def sample(self, noise, label, truncation=1., batch_size=8):
    noise = np.asarray(noise)
    label = np.asarray(label)
    num = noise.shape[0]
    if len(label.shape) == 0:
      label = np.asarray([label] * num)
    if label.shape[0] != num:
      raise ValueError('Got # noise samples ({}) != # label samples ({})'
                       .format(noise.shape[0], label.shape[0]))
    label = self.one_hot_if_needed(label)
    ims = []
    for batch_start in xrange(0, num, batch_size):
      s = slice(batch_start, min(num, batch_start + batch_size))
      feed_dict = {self.input_z: noise[s], self.input_y: label[s],
                   self.input_trunc: truncation}
      ims.append(self.sess.run(self.output, feed_dict=feed_dict))
    ims = np.concatenate(ims, axis=0)
    assert ims.shape[0] == num
    ims = np.clip(((ims + 1) / 2.0) * 256, 0, 255)
    ims = np.uint8(ims)
    return ims

  def generate_video(self, output_dir, starting_category=None, truncation=0.2,
                     video_file_name=None, noise_seed=0, batch_size=100):
    starting_category = (
        np.random.randint(
            len(categories.ALL_CATEGORIES)) if starting_category is None
        else starting_category)
    images_dir = os.path.join(output_dir, 'images')
    if not tf.io.gfile.exists(images_dir):
      tf.gfile.MakeDirs(images_dir)
    num_frames = len(self.running_mean_tv_diffs)
    category_idx = 0
    for start_index in range(0, num_frames, batch_size):
      if start_index == 0:
        z = truncated_z_sample(1, self.dim_z, truncation=truncation, seed=noise_seed)
        if category_idx < len(self.selected_categories):
          starting_category = self.one_hot(self.selected_categories[category_idx])
          category_idx += 1
        else:
          starting_category = self.one_hot(
                  [int(starting_category.split(')')[0])])
        y = starting_category
        starting_tv_diff = self.running_mean_tv_diffs[0]
        ending_tv_diff = self.running_mean_tv_diffs[self.inflection_points[0]]
        inflection_idx = 0
        if category_idx < len(self.selected_categories):
          target_category = self.one_hot(self.selected_categories[category_idx])
          category_idx += 1
        else:
          target_category = self.get_new_target_category()
      else:
        z = truncated_z_sample(1, self.dim_z, truncation=truncation, seed=noise_seed)
        alpha = self.alphas[start_index]
        y = (1 - alpha) * starting_category + alpha * target_category
        if start_index == self.inflection_points[inflection_idx]:
          # Get new target category.
          starting_category = target_category
          if category_idx < len(self.selected_categories):
            target_category = self.one_hot(self.selected_categories[category_idx])
            category_idx += 1
          else:
            target_category = self.get_new_target_category()
          starting_tv_diff = ending_tv_diff
          inflection_idx += 1
          ending_tv_diff = self.running_mean_tv_diffs[inflection_idx]
      end_index = min(start_index + batch_size,
                      min(len(self.running_mean_tv_diffs), len(self.alphas)))
      for i in range(start_index, end_index):
        alpha = self.alphas[i]
        z = np.append(z, truncated_z_sample(1, self.dim_z, truncation, noise_seed), axis=0)
        y = np.append(
            y, (1 - alpha) * starting_category + alpha * target_category,
            axis=0)
        if i == self.inflection_points[inflection_idx]:
          # Get new category.
          starting_category = target_category
          if category_idx < len(self.selected_categories):
            target_category = self.one_hot(self.selected_categories[category_idx])
            category_idx += 1
          else:
            target_category = self.get_new_target_category()
          starting_tv_diff = ending_tv_diff
          inflection_idx += 1
          ending_tv_diff = self.running_mean_tv_diffs[inflection_idx]
    
      ims = self.sample(z, y, truncation=truncation)
      _ = plt.xticks([])
      _ = plt.yticks([])
      for i in range(ims.shape[0]):
        plt.imsave(os.path.join(images_dir, '{:06d}'.format(i + start_index)),
                   ims[i])

    # Compile images into a video.
    video_file_name = (
            'video.avi' if video_file_name is None else video_file_name)
    destination_path = os.path.join(output_dir, video_file_name)
    fps = self.framerate / 128.
    files = [f for f in os.listdir(images_dir)]
    files.sort()
    vid = None
    for i in range(len(files)):
      filename = os.path.join(images_dir, files[i])
      img = cv2.imread(filename)
      if vid is None:
        height, width, _ = img.shape
        size = (width, height)
        vid = cv2.VideoWriter(destination_path,
                              cv2.VideoWriter_fourcc(*'MJPG'), fps,
                              (width, height))
      vid.write(img)
    vid.release()
