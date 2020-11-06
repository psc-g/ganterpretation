# ganterpretation
Code for making #GANterpretations is in the `src/` subdirectory.

A sample command for running by specifying an initial set of categories:

```
python src/run_ganterpreter.py --verbose \
  --wav_path=${WAV_PATH} \
  --output_dir=${OUTPUT_DIR} \
  --inflection_threshold=1e-2 \
  --video_file_name=${VIDEO_FILENAME}.avi \
  --selected_categories=419,419,419,107,617,127,730,3
```
