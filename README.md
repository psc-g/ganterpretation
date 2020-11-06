# GANterpretation
This is a repo that allows you to make a video from an audio file, where the video is made up of images generated by a GAN.
You can read more about it in my [paper](https://github.com/psc-g/ganterpretation/blob/master/ganterpretations.pdf), which was accepted to the [4th Workshop on Machine Learning for Creativity and Design at NeurIPS 2020](https://neurips2020creativity.github.io/).

## Credit attribution

If you tweet things you've generated with this, please use the hashtag `#GANterpretation` so I can see the awesome things you do!

To cite this work, please use:

```
@inproceedings{castro20ganterpretations,
  author    = {Pablo Samuel Castro},
  title     = {GANterpretations},
  year      = {2020},
  booktitle = {4th Workshop on Machine Learning for Creativity and Design at NeurIPS 2020},
}
```

## Samples

Here are some samples of this process:
-  [Bachbird](https://twitter.com/pcastr/status/1181767820834721792) (a piano arrangement I made merging The Beatles' Blackbird with J.S. Bach's Prelude in C# major)
-  [The story of GANdy](https://twitter.com/pcastr/status/1213296573804941312)
-  [Latent Voyage](https://twitter.com/pcastr/status/1197373969474736129)
-  [Modern Primates](https://twitter.com/pcastr/status/1197517036211097601)
-  [Zappa talking about music videos](https://twitter.com/pcastr/status/1182227164843958272)
-  [one small step for man, one GAN leap for manking](https://twitter.com/pcastr/status/1217833237092950017)


## Instructions

Code for making #GANterpretations is in the `src/` subdirectory.

For your convenience, [here](https://psc-g.github.io/ganterpretation/all_samples.html) is a website where you can scan through samples for the 1000 categories in BigGAN.

A sample command for running by specifying an initial set of categories:

```
python src/run_ganterpreter.py --verbose \
  --wav_path=${WAV_PATH} \
  --output_dir=${OUTPUT_DIR} \
  --inflection_threshold=1e-2 \
  --video_file_name=${VIDEO_FILENAME}.avi \
  --selected_categories=419,419,419,107,617,127,730,3
```
