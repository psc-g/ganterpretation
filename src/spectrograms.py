from scipy.io import wavfile
import numpy as np

def running_mean(x, N):
  """Compute running mean.
  """
  cumsum = np.cumsum(np.insert(x, 0, 0)) 
  return (cumsum[N:] - cumsum[:-N]) / float(N)

  
def get_inflection_points(arr, threshold):
  """Add inflection points.
  """
  direction = np.sign(arr[1:] - arr[:-1])
  assert len(arr) - 1 == len(direction)
  last_val = arr[0]
  curr_dir = 0
  inflection_points = []
  for i in range(len(direction)):
    if direction[i] != curr_dir:
      if abs(arr[i + 1] - last_val) > threshold:
        inflection_points.append(i + 1)
        last_val = arr[i + 1]
    curr_dir = direction[i]
  inflection_points.append(len(direction))
  return np.array(inflection_points)


def get_alpha(start, end, current, iteration):
  """Get alpha value for interpolations.
  """
  if start == end:
    return 0.
  numerator = abs(current - start)
  denominator = abs(end - start)
  return numerator / denominator


def get_spectrogram(framerate, sounddata, channel_to_use='Left',
                    inflection_threshold=0.035, verbose=False):
  channel = 0 if channel_to_use == 'Left' else 1
  spectrum, _, _, _ = plt.specgram(sounddata[channel], Fs=framerate)
  tv_diffs = [np.mean(np.abs(spectrum[:, i] - spectrum[:, i + 1])) for i in range(spectrum.shape[1] - 1)]
  tv_diffs /= max(tv_diffs)
  running_mean_tv_diffs = running_mean(tv_diffs, 256)
  running_mean_tv_diffs = np.append(np.array([running_mean_tv_diffs[0]] * 256),
                                    running_mean_tv_diffs)
  inflection_points = get_inflection_points(running_mean_tv_diffs,
                                            inflection_threshold)
  alphas = []
  next_inflection_idx = 0
  starting_tv = running_mean_tv_diffs[0]
  ending_tv = running_mean_tv_diffs[inflection_points[next_inflection_idx]]
  for i in range(1, len(running_mean_tv_diffs)):
    alphas.append(get_alpha(starting_tv, ending_tv, running_mean_tv_diffs[i], i))
    if i == inflection_points[next_inflection_idx]:
      starting_tv = ending_tv
      next_inflection_idx += 1
      try:
        ending_tv = running_mean_tv_diffs[inflection_points[next_inflection_idx]]
      except IndexError:
        pass
  alphas.append(1.0)
  if verbose:
    print('Number of inflection points: {}'.format(len(inflection_points)))
  return running_mean_tv_diffs, inflection_points, alphas
