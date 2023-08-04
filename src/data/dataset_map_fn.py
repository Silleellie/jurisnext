import random


def sample_sequence(batch):
    assert len(batch["text_sequence"]) == len(batch["title_sequence"])

    # a sequence has at least 1 data point, but it can have more depending on the length of the sequence
    # We must ensure that at least an element can be used as test set
    # in the "sliding_training_size" is included the target item
    sliding_size = random.randint(1, len(batch["text_sequence"]) - 1)

    # TO DO: consider starting always from the initial paragraph,
    # rather than varying the starting point of the seq
    start_index = random.randint(0, len(batch["text_sequence"]) - sliding_size - 1)
    end_index = start_index + sliding_size

    return {
        "case_id": batch["case_id"],
        "input_text_sequence": batch["text_sequence"][start_index:end_index],
        "input_title_sequence": batch["title_sequence"][start_index:end_index],
        "input_keywords_sequence": batch["rel_keywords_sequence"][start_index:end_index],
        "immediate_next_text": batch["text_sequence"][end_index],
        "immediate_next_title": batch["title_sequence"][end_index],
        "immediate_next_rel_keywords": batch["rel_keywords_sequence"][end_index]
    }
