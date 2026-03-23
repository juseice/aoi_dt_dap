def estimate_aoi(sense_delay, queue_delay, compute_delay, response_delay):
    return sense_delay + queue_delay + compute_delay + response_delay


def compute_aoi(arrival_time, generation_time):
    return arrival_time - generation_time
