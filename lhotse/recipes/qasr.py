"""
QASR is a large transcribed Arabic speech corpus with around 2,000 hours with multi-layer annotation, in multi-dialect and code-switching speech. The data is crawled from the Aljazeera news channel with lightly supervised transcriptions and linguistically motivated segmentation. QASR is suitable for training and evaluating speech recognition systems, acoustics- and/or linguistics-based Arabic dialect identification, punctuation restoration, speaker identification, speaker linking, and potentially other natural language processing modules for spoken data.
"""
from itertools import chain
from logging import info
from os import path, system
from pathlib import Path
from re import match, sub
from shutil import copy
from string import punctuation
from typing import Dict, Union

from lhotse import (
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    fix_manifests,
    validate_recordings_and_supervisions,
)
from lhotse.kaldi import load_kaldi_data_dir
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.utils import Pathlike, check_and_rglob, is_module_available, recursion_limit


def download_qasr(
    target_dir: Pathlike = ".",
) -> None:
    """
    Download and untar the dataset.

    NOTE: This function just returns with a message since MGB2 is not available
    for direct download.

    :param target_dir: Pathlike, the path of the dir to storage the dataset.
    """
    info(
        "MGB2 is not available for direct download. Please fill out the form"
        "at https://arabicspeech.org/qasr to download the corpus."
    )


def prepare_qasr(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    text_cleaning: bool = True,
    num_jobs: int = 1,
    mer_thresh: int = 80,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.

    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param text_cleaning: Bool, if True, basic text cleaning is performed (similar to ESPNet recipe).
    :param num_jobs: int, the number of jobs to use for parallel processing.
    :param mer_thresh: int, filter out segments based on mer (Match Error Rate)
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.

    .. note::
        Unlike other recipes, output_dir is not Optional here because we write the manifests
        to the output directory while processing to avoid OOM issues, since it is a large dataset.

    .. caution::
        The `text_cleaning` option removes all punctuation and diacritics.
    """

    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    dataset_parts = ["test", "train_20210109", "dev"]
    manifests = {}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Maybe the manifests already exist: we can read them and save a bit of preparation time.
        manifests = read_manifests_if_cached(
            dataset_parts=dataset_parts,
            output_dir=output_dir,
            prefix="qasr",
            suffix="jsonl.gz",
            lazy=True,
        )

    for part in dataset_parts:
        info(f"Processing QASR subset: {part}")
        if manifests_exist(
            part=part, output_dir=output_dir, prefix="qasr", suffix="jsonl.gz"
        ):
            info(f"QASR subset: {part} already prepared - skipping.")
            continue

        # Read the recordings and write them into manifest. We additionally store the
        # duration of the recordings in a dict which will be used later to create the
        # supervisions.

        output_dir = Path(output_dir)
        corpus_dir = Path(corpus_dir)
        filt_ids = None
        if part == "test":
            
            supervisions_list = read_text(corpus_dir /'release'/ part/ 'text.verbatim.nonoverlap')
            recordings = RecordingSet.from_dir(
                (corpus_dir /'release'/ part / "wav"), pattern="*.wav", num_jobs=num_jobs
            )
            xml_paths = check_and_rglob(
                path.join(corpus_dir, 'release', part, "xml"), "*.xml"
            )
            
            supervisions = SupervisionSet.from_segments(supervisions_list).filter(
            lambda s: s.duration > 0.0)
            assert (
                    len(supervisions) == 5366
                ), f"Expected 5366 supervisions for test, found {len(supervisions)}"

        elif part == "dev":
            supervisions_list = read_text(corpus_dir /'release'/ part/ 'text.verbatim.nonoverlap')
            recordings = RecordingSet.from_dir(
                (corpus_dir /'release'/ part / "wav"), pattern="*.wav", num_jobs=num_jobs
            )
            xml_paths = check_and_rglob(
                path.join(corpus_dir, 'release', part, "xml"), "*.xml"
            )
            supervisions = SupervisionSet.from_segments(supervisions_list).filter(
            lambda s: s.duration > 0.0)
            assert (
                    len(supervisions) == 5002
                ), f"Expected 5002 supervisions for dev, found {len(supervisions)}"
        elif "train" in part:
            recordings = RecordingSet.from_dir(
                (corpus_dir / "wav"), pattern="*.wav", num_jobs=num_jobs
            )

            xml_paths = check_and_rglob(
                path.join(corpus_dir, 'release', part, "xml"), "*.xml"
            )
            # Read supervisions and write them to manifest
            with recursion_limit(5000):
                supervisions_list = list(
                    chain.from_iterable(
                        [make_supervisions(p, mer_thresh) for p in xml_paths]
                    )
                )
            supervisions = SupervisionSet.from_segments(supervisions_list).filter(
            lambda s: s.duration > 0.0)
            assert (
                len(supervisions) == 1592226
            ), f"Expected 375103 supervisions for train, found {len(supervisions)}"

        if text_cleaning is True:
            supervisions = supervisions.transform_text(cleaning)
        recordings, supervisions = fix_manifests(recordings, supervisions)
        validate_recordings_and_supervisions(recordings, supervisions)

        # saving recordings and supervisions
        recordings.to_file((output_dir / f"qasr_recordings_{part}.jsonl.gz"))
        supervisions.to_file((output_dir / f"qasr_supervisions_{part}.jsonl.gz"))

        manifests[part] = {
            "recordings": recordings,
            "supervisions": supervisions,
        }
    return manifests


def read_text(file):
    segments = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line1, text = line.strip().split()[0], " ".join(line.strip().split()[1:])
            rec_id, spk, time = line1.split('_')
            spk = int(spk.split('-')[1])
            s_time1, e_time1 = time.replace('seg-', '').split(':')
            s_time, e_time = float(int(s_time1)/100), float(int(e_time1)/100)
            segments.append(
                SupervisionSegment(
                    id=rec_id + "_" + str(spk) +"_" +s_time1 +"_"+e_time1,
                    recording_id=rec_id,
                    start=s_time,
                    duration=round(e_time - s_time, 10),
                    channel=0,
                    text=text,
                    language='ar',
                    speaker=spk,
                )
            )
    return segments


def remove_diacritics(text: str) -> str:
    # https://unicode-table.com/en/blocks/arabic/
    return sub(r"[\u064B-\u0652\u06D4\u0670\u0674\u06D5-\u06ED]+", "", text)


def remove_punctuations(text: str) -> str:
    """This function  removes all punctuations except the verbatim"""

    arabic_punctuations = """﴿﴾`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ"""
    english_punctuations = punctuation
    # remove all non verbatim punctuations
    all_punctuations = set(arabic_punctuations + english_punctuations)

    for p in all_punctuations:
        if p in text:
            text = text.replace(p, " ")
    return text


def remove_non_alphanumeric(text: str) -> str:
    text = text.lower()
    return sub(r"[^\u0600-\u06FF\s\da-z]+", "", text)


def remove_single_char_word(text: str) -> str:
    """
    Remove single character word from text
    Example: I am in a a home for two years => am in home for two years
    Args:
            text (str): text
    Returns:
            (str): text with single char removed
    """
    words = text.split()

    filter_words = [word for word in words if len(word) > 1 or word.isnumeric()]
    return " ".join(filter_words)


def east_to_west_num(text: str) -> str:
    eastern_to_western = {
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
        "٪": "%",
        "_": " ",
        "ڤ": "ف",
        "|": " ",
    }
    trans_string = str.maketrans(eastern_to_western)
    return text.translate(trans_string)


def remove_extra_space(text: str) -> str:
    text = sub(r"\s+", " ", text)
    text = sub(r"\s+\.\s+", ".", text)
    return text


def cleaning(text: str) -> str:
    text = remove_punctuations(text)
    text = east_to_west_num(text)
    text = remove_diacritics(text)
    text = remove_non_alphanumeric(text)
    text = remove_single_char_word(text)
    text = remove_extra_space(text)
    return text


def make_supervisions(xml_path: str, mer_thresh: int) -> None:
    if not is_module_available("bs4"):
        raise ValueError(
            "To prepare QASR data, please 'pip install beautifulsoup4' first."
        )
    from bs4 import BeautifulSoup

    xml_handle = open(xml_path, "r")
    soup = BeautifulSoup(xml_handle, "xml")
    return [
        SupervisionSegment(
            id=segment["id"].split("_utt")[0] + "_" + str(int(match(r"\w+speaker(\d+)\w+", segment["who"]).group(1))) + '_' + str(int(float(segment["starttime"])*100)).zfill(7) + "_" + str(int(float(segment["endtime"])*100)).zfill(7),
            recording_id=segment["id"].split("_utt")[0].replace("_", "-"),
            start=float(segment["starttime"]),
            duration=round(
                float(segment["endtime"]) - float(segment["starttime"]), ndigits=8
            ),
            channel=0,
            text=" ".join(
                [
                    element.string
                    for element in segment.find_all("element")
                    if element.string is not None
                ]
            ),
            language="ar",
            speaker=int(match(r"\w+speaker(\d+)\w+", segment["who"]).group(1)),
        )
        for segment in soup.find_all("segment")
        if mer_thresh is None or float(segment["WMER"]) <= mer_thresh
    ]


def main():
    corpus_dir = Path("/alt/arabic-speech-web/mgb2.1")
    output_dir = Path("/alt-arabic/speech/amir/qasr")
    prepare_qasr(corpus_dir=corpus_dir, output_dir=output_dir, num_jobs = 8)

if __name__ == '__main__':
    main()