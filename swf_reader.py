import csv
import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import pytz

# NASA StartTime: Fri Oct 01 00:00:03 PDT 1993 EndTime:   Fri Dec 31 23:03:45 PST 1993
# init_time = datetime.datetime(year=1993, month=10,day=1,hour=0,minute=0, second=3)
# Create a timezone-aware datetime for the initial time
init_time = pytz.timezone("US/Pacific").localize(
    datetime.datetime(1993, 10, 1, 0, 0, 3)
)


def process_lines(unprocessed_lines):
    processed_lines = []
    for line in unprocessed_lines:
        if line.startswith(";"):
            continue
        else:
            processed_lines.append(line.split())
    return processed_lines


def detailed_time(lines):
    for line in lines:
        submit_time = line[1]
        curr_time = init_time + datetime.timedelta(seconds=int(submit_time))
        curr_time = curr_time.astimezone(pytz.timezone("US/Pacific"))
        line.insert(2, curr_time.year)
        line.insert(3, curr_time.month)
        line.insert(4, curr_time.day)
        line.insert(
            5, curr_time.weekday() + 1
        )  # weekday() returns 0 for Monday and 6 for Sunday, so we add 1 to get the correct day of the week (1 for Monday and 7 for Sunday, as in the dataset description
        line.insert(6, curr_time.hour)
        line.insert(7, curr_time.minute)
        line.insert(8, curr_time.second)
    return lines


def write_csv_file(filename, processed_lines, fields):
    with open(filename, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=";")
        csvwriter.writerow(fields)
        processed_lines = detailed_time(processed_lines)
        csvwriter.writerows(processed_lines)


def unix_to_date_info(unix_time):
    # Start time is the number of seconds since Fri Oct 01 00:00:03 PDT 1993
    # Returns the hour, minute, second, day, month, year and week day in list
    # Week day 0 is Sunday and week day 6 is Saturday
    return datetime.datetime.fromtimestamp(int(unix_time) + 749430003).strftime(
        "%H;%M;%S;%d;%m;%Y;%w"
    )


def remove_collumns(csv_filename):
    f = pd.read_csv(csv_filename, delimiter=";")
    f = f.loc[:, (f != -1).any(axis=0)]
    f.to_csv(csv_filename, sep=";", index=False)


def node_usage_per_timeframe(processed_lines, time_factor):
    start_time = int(processed_lines[0][1])
    end_time = int(processed_lines[-1][1]) + int(processed_lines[-1][10])
    end_timeframe = (int(end_time) - int(start_time)) // time_factor + 1
    timeframe = np.zeros(end_timeframe)
    for line in processed_lines:
        job_start_timeframe = (int(line[1]) - start_time) // time_factor
        job_end_timeframe = (int(line[1]) + int(line[10]) - start_time) // time_factor
        for i in range(job_start_timeframe, job_end_timeframe + 1):
            if i >= end_timeframe:
                print("Error: timeframe out of bounds")
                break
            timeframe[i] += int(line[11])
    for i in range(end_timeframe):
        if timeframe[i] > 128:
            timeframe[i] = 128
    return timeframe


def main():
    current_path = Path(__file__).parent.absolute()
    csv_filename = str(current_path) + "\\Csv_data\\mei_nasa_1993_cleaned.csv"
    # csv_filename = str(current_path) + "\\Csv_data\\mei_unigala_1993.csv"

    source_swf_file = (
        str(current_path)
        # + "\\Datasets\\UniLu-Gaia-2014-2.swf\\UniLu-Gaia-2014-2.swf"
        + "\\Datasets\\NASA-iPSC-1993-3.1-cln.swf\\NASA-iPSC-1993-3.1-cln.swf"
    )

    # user and command_num are incremental
    # max_nodes are 128
    # start_time is the number of seconds since Fri Oct 01 00:00:03 PDT 1993
    # queue_num is 0 for interactive jobs and 1 for batch jobs
    # user_group is 1 for normal users and 2 for system personnel
    # number of nodes allocated to the job is always a power of 2
    fields_nasa_1993 = [
        "job_number",
        "submit_time",
        "year",
        "month",
        "day",
        "day_of_week",
        "hour",
        "minute",
        "second",
        "wait_time",
        "runtime",
        "nodes_alloc",
        "cpu_used",
        "mem_used",
        "proc_req",
        "time_req",
        "mem_req",
        "status",
        "user",
        "user_group",
        "executable_num",
        "queue_num",
        "partition",
        "prev_job",
        "think_time",
    ]

    swf_file = open(source_swf_file, "r")
    unprocessed_lines = swf_file.readlines()
    processed_lines = process_lines(unprocessed_lines)
    # print(processed_lines)
    for line in processed_lines:
        date_time = unix_to_date_info(line[1])
        for element in date_time.split(";"):
            processed_lines[processed_lines.index(line)].append(element)
    write_csv_file(csv_filename, processed_lines, fields_nasa_1993)
    remove_collumns(csv_filename)
    swf_file.close()

    nodes_used_per_hour = np.zeros(24)
    usage_1_sec_timeframe = node_usage_per_timeframe(processed_lines, 1)
    print(usage_1_sec_timeframe)
    for i, nodes in enumerate(usage_1_sec_timeframe):
        hours_since_start = int(i) // (3600)
        hour = hours_since_start % 24
        nodes_used_per_hour[hour] += nodes

    nodes_used_per_hour = nodes_used_per_hour / (3600 * 92)
    hours = np.arange(24)
    plt.plot(hours, nodes_used_per_hour)
    plt.title("Node usage per hour")
    plt.xlabel("Hour")
    plt.ylabel("Nodes used")
    plt.show()

    nodes_used_per_weekday = np.zeros(7)
    usage_1_sec_timeframe = node_usage_per_timeframe(processed_lines, 1)
    for i, nodes in enumerate(usage_1_sec_timeframe):
        days_since_start = int(i) // (3600 * 24)
        weekday = (days_since_start + 4) % 7
        nodes_used_per_weekday[weekday] += nodes

    nodes_used_per_weekday = nodes_used_per_weekday / (3600 * 24 * 13)
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    plt.plot(weekdays, nodes_used_per_weekday)
    plt.title("Node usage per day of the week")
    plt.xlabel("Day of the week")
    plt.ylabel("Nodes used")
    plt.show()

    total_number_of_timeframes = len(usage_1_sec_timeframe)
    timeframes_with_128_nodes = 0
    for value in usage_1_sec_timeframe:
        if value >= 128:
            timeframes_with_128_nodes += 1

    print("Total number of timeframes: ", total_number_of_timeframes)
    print("Timeframes with 128 nodes: ", timeframes_with_128_nodes)
    print(
        "Percentage of timeframes with 128 nodes: ",
        timeframes_with_128_nodes / total_number_of_timeframes * 100,
        "%",
    )


if __name__ == "__main__":
    main()
