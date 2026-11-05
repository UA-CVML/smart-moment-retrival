import os
import re
import ast

import torch
from torch.cuda.amp import autocast as autocast
import wandb
from torch.nn import functional as F

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def validate_and_correct(pred):
    match = re.match(r"^\[\[(\d+,\s*\d+\](,\s*\[\d+,\s*\d+\])*)\]$", pred)
    if match:
        return pred
    else:
        corrected_pred = re.findall(r"\[\d+,\s*\d+\]", pred)
        
        if corrected_pred:
            return "[" + ", ".join(corrected_pred) + "]"
        else:
            return "[[-1, -1]]"

def post_process(pred):
    """Post process predicted output to be in the format of moments, i.e. [[0, 1], [4, 7]].
        - if no comma, i.e. " " → add comma, i.e. ", "
        - if t_start > t_end → swap them
        - if two comma: ",," → ","
    Args:
        pred (str): predicted output with potential errors, e.g. "[[0, 1], [4, 7]]"
    Returns:
        str: post processed predicted output, e.g. "[[0, 1], [4, 7]]"
    """
    pred = pred.split("</s>")[0]

    pred = validate_and_correct(pred)

    pred = pred[1:-1]

    windows = re.split(r"\s+(?=\[)", pred)
    output = []

    for window in windows:

        window = re.sub(r",+$", "", window)

        window = re.sub(r"(\d) (\d)", r"\1, \2", window)

        window = re.sub(r",+", ",", window)

        numbers = re.findall(r"\d+", window)

        if len(numbers) == 2:
            t_start, t_end = numbers
            window = "[" + t_start + ", " + t_end + "]"
            if float(t_start) > float(t_end):
                window = "[" + t_end + ", " + t_start + "]"

            output.append(window)
    output = "[" + ", ".join(output) + "]"
    return output

def post_process_TAL(pred):
    """Post process predicted output to be in the format of moments with labels, i.e. [[0, 1, "label"], [4, 7, "label"]].
        - if no comma, i.e. " " → add comma, i.e. ", "
        - if t_start > t_end → swap them
        - if two comma: ",," → ","
    Args:
        pred (str): predicted output with potential errors, e.g. "[[0, 1, "label"], [4, 7, "label"]]"
    Returns:
        str: post processed predicted output, e.g. "[[0, 1, "label"], [4, 7, "label"]]"
    """

    pred = pred.split("</s>")[0]

    pred = re.sub(r",+\]", "]", pred)

    if not re.match(r"\[\[.*\]\]", pred):
        return "[[-1, -1, -1]]"

    pred = pred[1:-1]

    windows = re.split(r"\s+(?=\[)", pred)

    output = []

    for window in windows:

        window = re.sub(r",+$", "", window)

        window = re.sub(r"(\d) (\d)", r"\1, \2", window)

        window = re.sub(r"(\d), (\d) (\w+)", r"\1, \2, \3", window)

        window = re.sub(r"(\d) (\d) (\w+)", r"\1, \2, \3", window)

        window = re.sub(r",+", ",", window)

        numbers = re.findall(r"\d+", window)

        text = re.sub(r"\d+", "", window)

        label = re.findall(r"\w+", text)
        if label == []:
            label = ['"No label"']

        if len(numbers) == 2:
            t_start, t_end = numbers
            if int(t_start) > int(t_end):
                window = "[" + t_end + ", " + t_start + ", '" + " ".join(label) + "']"

        else:
            return "[[-1, -1, -1]]"

        output.append(window)

    output = "[" + ", ".join(output) + "]"

    return output

def format_wandb_log_images_and_predictions(
    samples,
    wandb_table_data,
    pred,
    video_prompt,
    post_process_fn=post_process,
    input_time_format=None,
    interleave_data=True,
    train_data=True,
):
    out = {}

    image, qid = samples["video"], samples["query_id"]
    b, t, c, h, w = image.size()
    query_prompt = samples["query_prompt"]
    answer = samples["relevant_windows"]

    idx = torch.randint(0, b, (1,)).item()
    frames = []
    for frame in image[idx]:
        frame = frame.cpu().numpy().transpose(1, 2, 0)
        frame = wandb.Image(frame)
        frames.append(frame)
    if interleave_data:
        query = video_prompt[idx] + "</vid>" + query_prompt[idx]
    else:
        query = video_prompt[idx] + "<frames> </vid>" + query_prompt[idx]

    pred = pred[idx]
    processed_pred = post_process_fn(pred)
    qid = qid[idx]
    answer = answer[idx]
    duration = samples["duration"][idx]

    if (
        input_time_format == "relative_integers"
        or input_time_format == "relative_floats"
    ):
        processed_pred = convert_to_absolute_time([processed_pred], [duration])

    wandb_table_data.append(
        [
            qid,
            frames,
            query,
            pred,
            processed_pred,
            answer,
            duration,
        ]
    )

    wandb_table = wandb.Table(
        columns=[
            "qid",
            "frames",
            "query",
            "pred",
            "processed_pred",
            "answer",
            "duration",
        ]
    )

    for row in wandb_table_data:
        wandb_table.add_data(*row)

    if train_data:
        out["Samples_during_training"] = wandb_table
    else:
        out["Samples_during_eval"] = wandb_table

    return out, wandb_table_data

def convert_to_absolute_time(prediction, duration, input_time_format):
    """Convert relative timestamps to absolute timestamps.
    Args:
        prediction (list(str)): A list of predicted moments. Each moment is a list of start and end time of the moment, e.g. "[[0.0, 0.5], [1.0, 2.0]]"
        duration (list): A list of video durations.
    Returns:
        list(str): A list strings with the predicted moments as absolute timestamps.
    """

    assert (
        input_time_format == "relative_integers"
        or input_time_format == "relative_floats"
    ), "This function is only used for relative timestamps"

    prediction = [moment_str_to_list(m) for m in prediction]

    absolute_prediction = []

    for pred, dur in zip(prediction, duration):

        if input_time_format == "relative_integers":
            absolute_prediction.append(
                [
                    (
                        [
                            round((float(start) / 100) * dur, 2),
                            round((float(end) / 100) * dur, 2),
                        ]
                        if start != -1 and end != -1
                        else [-1, -1]
                    )
                    for start, end in pred
                ]
            )
        else:
            absolute_prediction.append(
                [
                    (
                        [round(float(start) * dur, 2), round(float(end) * dur, 2)]
                        if start != -1 and end != -1
                        else [-1, -1]
                    )
                    for start, end in pred
                ]
            )

    absolute_prediction = [str(m) for m in absolute_prediction]

    return absolute_prediction

def moment_str_to_list(m):
    """Convert a string of moments to a list of moments.
    If predicted string is not a list, it means that the model has not yet learned to predict the right format.
    In that case, we return [[-1, -1]] to represent an error.
    This will then lead to an IoU of 0.
    Args:
        m (str): a string of moments, e.g. "[[0, 1], [4, 7]]"
    Returns:
        list: a list of moments, e.g. [[0, 1], [4, 7]]
    """
    if m == "[[-1, -1]]":
        return [[-1, -1]]

    if not re.match(r"\[\[.*\]\]", m):
        return [[-1, -1]]

    try:
        _m = ast.literal_eval(m)
    except:
        return [[-1, -1]]

    if not isinstance(_m, list):

        return [[-1, -1]]

    for i in range(len(_m)):
        if isinstance(i, int):
            _m[i] = [-1, -1]
        if len(_m[i]) != 2:

            _m[i] = [-1, -1]

    return _m

def tal_str_to_list(m):
    """Convert a string of moments and a class label to a list of moments and labels.
    If predicted string is not a list, it means that the model has not yet learned to predict the right format.
    In that case, we return [[-1, -1]] to represent an error.
    This will then lead to an IoU of 0.
    Args:
        m (str): a string of moments, e.g. "[[0, 1, "label"], [4, 7, "label"]]"
    Returns:
        list: a list of moments, e.g. [[0, 1, "label"], [4, 7, "label"]]
    """
    if m == "[[-1, -1, -1]]":
        return [[-1, -1, -1]]

    if not re.match(r"\[\[.*\]\]", m):
        return [[-1, -1, -1]]

    try:
        _m = ast.literal_eval(m)
    except:
        return [[-1, -1, -1]]

    if not isinstance(_m, list):

        return [[-1, -1, -1]]

    for i in range(len(_m)):

        if len(_m[i]) != 3:

            _m[i] = [-1, -1, -1]

    return _m

def get_timestamps_as_seconds_integers(
    timestamps, durations, annoying_numbers_replacement_dict=None
):

    new_video_prompts = []
    new_timestamps = []
    new_durations = []

    for t, d in zip(timestamps, durations):
        duration = d.item()

        _video_prompt = ">".join(
            (
                str(int(round(timestamp.item())))
                if round(timestamp.item())
                not in annoying_numbers_replacement_dict.keys()
                else str(annoying_numbers_replacement_dict[round(timestamp.item())])
            )
            for timestamp in t
        )

        duration = (
            round(duration)
            if round(duration) not in annoying_numbers_replacement_dict.keys()
            else annoying_numbers_replacement_dict[round(duration)]
        )

        _video_prompt = ">" + _video_prompt + ">" + str(duration)
        new_video_prompts.append(_video_prompt)

        new_timestamps.append(
            torch.tensor(
                [
                    int(
                        round(timestamp.item())
                        if round(timestamp.item())
                        not in annoying_numbers_replacement_dict.keys()
                        else annoying_numbers_replacement_dict[round(timestamp.item())]
                    )
                    for timestamp in t
                ]
            )
        )
        new_durations.append(duration)
    return new_timestamps, new_durations, new_video_prompts

def get_timestamps_as_relative_integers(
    timestamps, durations, annoying_numbers_replacement_dict=None
):

    new_video_prompts = []
    new_timestamps = []

    for t, d in zip(timestamps, durations):

        duration = d.item()

        _video_prompt = ">".join(
            str(int(round((timestamp.item() / duration), 2) * 100)) for timestamp in t
        )

        _video_prompt += ">" + str(round(duration))
        new_video_prompts.append(_video_prompt)

        new_timestamps.append(
            torch.tensor(
                [int(round((timestamp.item() / duration), 2) * 100) for timestamp in t]
            )
        )

    return new_timestamps, durations, new_video_prompts

def get_timestamps_as_seconds_floats(
    timestamps, durations, annoying_numbers_replacement_dict=None
):

    new_video_prompts = []
    new_timestamps = []

    for t, d in zip(timestamps, durations):

        duration = d.item()

        _video_prompt = ">".join(str(round(timestamp.item(), 2)) for timestamp in t)

        _video_prompt += ">" + str(round(duration))
        new_video_prompts.append(_video_prompt)

        new_timestamps.append(
            torch.tensor([round(timestamp.item(), 2) for timestamp in t])
        )

    return new_timestamps, durations, new_video_prompts

def get_timestamps_as_relative_floats(
    timestamps, durations, annoying_numbers_replacement_dict=None
):

    new_video_prompts = []
    new_timestamps = []

    for t, d in zip(timestamps, durations):
        duration = d.item()

        _video_prompt = ">".join(
            str(round((timestamp.item() / duration), 2)) for timestamp in t[:-1]
        )

        _video_prompt += ">" + str(round(duration))
        new_video_prompts.append(_video_prompt)

        new_timestamps.append(
            torch.tensor(
                [round((timestamp.item() / duration), 2) for timestamp in t]
                + [round(duration)]
            )
        )

    return new_timestamps, durations, new_video_prompts

def get_timestamps_as_framenumbers(
    timestamps, durations, annoying_numbers_replacement_dict=None
):

    new_video_prompts = []
    new_timestamps = []
    for t, d in zip(timestamps, durations):
        _video_prompt = ">".join(str(i) for i in range(len(t)))
        _video_prompt += ">" + d.item()
        new_video_prompts.append(_video_prompt)

        new_timestamps.append(torch.tensor([i for i in range(len(t))]))

    return new_timestamps, durations, new_video_prompts
