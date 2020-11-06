from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl import app
from absl import flags

import ganterpreter

flags.DEFINE_string('output_dir', None, 'Directory where to store output.')
flags.DEFINE_string('wav_path', None, 'Path to wav file to use.')
flags.DEFINE_string('video_file_name', None,
                    'Name of video file, defaults to "video.avi"')
flags.DEFINE_string('model_type', 'biggan-512',
                    'BigGAN model type to load (biggan-{128, 256, 512})')
flags.DEFINE_list('selected_categories', [],
                      'Manually specified categories to use for interpolation. '
                      'Missing categories will be assigned randomly.')
flags.DEFINE_float('inflection_threshold', 0.035,
                   'Threshold on FFT TotalVariation changes to set '
                   'inflection points.')
flags.DEFINE_bool('verbose', False, 'Whether to print verbose messages.')

FLAGS = flags.FLAGS


def main(unused_argv):
  """Main method."""
  selected_categories = [int(x) for x in FLAGS.selected_categories]
  gandy = ganterpreter.GANterpreter(
      model_type=FLAGS.model_type,
      selected_categories=selected_categories,
      verbose=FLAGS.verbose)
  gandy.load_wav_file(FLAGS.wav_path, verbose=FLAGS.verbose)
  gandy.compute_spectrogram(inflection_threshold=FLAGS.inflection_threshold,
                          verbose=FLAGS.verbose)
  gandy.fill_selected_categories()
  gandy.generate_video(FLAGS.output_dir,
                     video_file_name=FLAGS.video_file_name)


if __name__ == '__main__':
  # flags.mark_flag_as_required('base_dir')
  app.run(main)
