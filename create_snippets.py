import os
import pandas as pd
from pydub import AudioSegment
import numpy as np
from tqdm import tqdm
import glob


def create_snippets(filename, audio_len, overlap_pct, max_padding_pct):
    """
    Creates audio snippets from long audio streams

    :param filename:
    :param audio_len: audio length of audio snippets in ms
    :param overlap_pct: overlap percentage between snippets
    :param max_padding_pct: maximum padding at the end of a snippet,
                            used if the last snippet in shorter than
                            the specified audio length
    """

    # Read labels file
    labels = pd.read_csv(
        os.path.join("audio", "original", f"{filename}.txt"),
        sep="\t",
        header=None,
        names=["start", "end", "label"],
    )
    # Replace / and convert to lowercase
    labels["label"] = labels["label"].str.replace("/", "_").str.lower()

    # Create folders for each label is they don't exist
    for label in labels["label"].unique():
        if not os.path.exists(os.path.join("audio", "snippets", label)):
            os.makedirs(os.path.join("audio", "snippets", label), exist_ok=True)

    # Load in audio file
    audio = AudioSegment.from_wav(os.path.join("audio", "original", f"{filename}.wav"))

    # Set counter for number of snippets
    counter = {"music": 0, "ads_other": 0}

    # Cut audio stream based on labels
    for index, row in tqdm(labels.iterrows()):
        start = row["start"] * 1000  # convert to ms
        end = row["end"] * 1000  # convert to ms
        clip = audio[start:end]

        # Create list with cutting points
        step = audio_len - (audio_len * overlap_pct)
        # Add one to make sure that last number is included as arange in non-inclusive
        slices = np.arange(0, len(clip) + 1, step)
        # Cut audio stream into segments
        segments = [clip[start:end] for start, end in zip(slices[:-1], slices[2:])]
        # Append last clip with length shorter than (audio_len - overlap)
        segments.append(clip[slices[-1] :])
        pad_len = audio_len - len(segments[-1])
        # Add silence to last clip if below pad length limit
        if pad_len <= audio_len * max_padding_pct:
            segments[-1] = segments[-1] + AudioSegment.silent(
                duration=pad_len, frame_rate=clip.frame_rate
            )
        # Else remove last item
        else:
            segments = segments[:-1]

        # Save all segments
        for segment in tqdm(segments):
            counter[row["label"]] += 1
            segment.export(
                os.path.join(
                    "audio", "snippets", row["label"], f"{filename}_{counter[row['label']]}.wav"
                ),
                format="wav",
            )


if __name__ == "__main__":
    # Max length of audio snippets in ms
    MAX_AUDIO_LEN = 1000
    # Min length of audio snippets as percentage of MAX_AUDIO_LEN
    MIN_AUDIO_PCT = 0.5
    # Overlap of segments as a percentage from MAX_AUDIO_LEN
    OVERLAP_PCT = 0.5

    filenames = [x for x in os.listdir("audio/original") if x.endswith(".wav")]
    for filename in tqdm(filenames):
        create_snippets(os.path.splitext(filename)[0], MAX_AUDIO_LEN, MIN_AUDIO_PCT, OVERLAP_PCT)

    # Create folders for data split
    labels = [x[1] for x in os.walk("audio/snippets")][0]
    splits = ["train", "validation", "test"]
    for split in splits:
        for label in labels:
            os.makedirs(os.path.join("audio", split, label), exist_ok=True)

    # Split files between training, test (and validation)
    for label in tqdm(labels, ascii=True):
        for filename in tqdm(
            [x for x in os.listdir(f"audio/snippets/{label}") if x.endswith(".wav")],
            ascii=True
        ):
            split = np.random.choice(splits, p=[0.7, 0.2, 0.1])
            os.rename(
                os.path.join("audio/snippets", label, filename),
                os.path.join("audio", split, label, filename),
            )

    # Remove empty folders
    for label in labels:
        os.removedirs(f"audio/snippets/{label}")

    # Get number of snippets
    metadata = pd.DataFrame({"path": glob.glob("audio/**/*.wav", recursive=True)})
    metadata["class"] = metadata.path.str.split("/").str[2]
    metadata["split"] = metadata.path.str.split("/").str[1]
    metadata = metadata[metadata.split != "original"]
    print("Number of snippets:")
    print(metadata.groupby(["split", "class"]).count())
    # Save metadata
    metadata.to_csv("metadata.csv", index=False)
