import numpy as np

dir_result = "/home/aeryen/Desktop/PycharmProjects/CNN_Class/runs/my_beer/full_loop/1481974264_120/"

count_file = open("/home/aeryen/Desktop/PycharmProjects/CNN_Class/data/beer_100k/test.count", "r")
review_file = open("/home/aeryen/Desktop/PycharmProjects/CNN_Class/data/beer_100k/test.txt", "r")
rating_file = open("/home/aeryen/Desktop/PycharmProjects/CNN_Class/data/beer_100k/test.rating", "r")

aspect_result = open(dir_result + "aspect_related_name.out", "r")
sent_score_file = open(dir_result + "aspect_sway.out", "r")
review_score_file = open(dir_result + "aspect_rating.out", "r")

count_lines = count_file.readlines()
review_lines = review_file.readlines()
rating_lines = rating_file.readlines()

rating_lines = [line.split() for line in rating_lines]
rating_lines = np.array(rating_lines).astype(float)

aspect_lines = aspect_result.readlines()
sent_score_lines = sent_score_file.readlines()
review_score_lines = review_score_file.readlines()

sent_score_lines = np.array(sent_score_lines).astype(float)
sent_score_lines = sent_score_lines / 2.0

review_score_lines = [line.split() for line in review_score_lines]
review_score_lines = np.array(review_score_lines).astype(float)
review_score_lines = review_score_lines / 2.0

output_file = open(dir_result + "formatted.out", "w")

review_count = len(count_lines)

global_index = 0
review_index = 0
for review_line_n in count_lines:
    line_n = int(review_line_n)
    output_file.write(str(rating_lines[review_index]).replace(" ", "\t") + "\n")
    output_file.write(str(review_score_lines[review_index]).replace(" ", "\t") + "\n")

    for line_index in range(line_n):
        output_file.write(aspect_lines[global_index].strip() + "\t")
        output_file.write(str(sent_score_lines[global_index]) + "\t")
        output_file.write(review_lines[global_index])
        # output_file.write("\n")
        global_index += 1

    output_file.write("\n")
    output_file.write("\n")
    review_index += 1

output_file.close()
