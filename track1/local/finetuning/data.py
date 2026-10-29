"""Read indexed training and validation audio."""

from robohead_baseline.track1.local.finetuning.audio import read_training_audio


def read_audio_record(row):
    return read_training_audio(row, verify=True)
