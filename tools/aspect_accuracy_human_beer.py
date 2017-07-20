from sklearn.metrics import classification_report

input_dir = "../runs/my_beer/full_loop/1481974264_120/"

predict_aspect_list = list(open(input_dir + "aspect_related_name.out", "r").readlines())
predict_aspect_list = [s.strip().lower() for s in predict_aspect_list if (len(s) > 0 and s != "\n")]
predict_aspect_list = ['none' if s == 'all' else s for s in predict_aspect_list]

aspect_set = set(predict_aspect_list)
print aspect_set

input_dir = "../data/beer_100k/"

file_yifan = list(open(input_dir + "test_fan.aspect", "r").readlines())
aspect_yifan = [s.strip() for s in file_yifan if (len(s) > 0 and s != "\n")]
aspect_yifan = [s.split()[0] for s in aspect_yifan]
# aspect_yifan = [s[0] for s in file_yifan]
# polar_yifan = [s[1] for s in file_yifan]

aspect_set = set(aspect_yifan)
print aspect_set

# file_fan = list(open(input_dir + "test_aspect_0.fan.aspect", "r").readlines())
# file_fan = [s.strip() for s in file_fan if (len(s) > 0 and s != "\n")]
# aspect_fan = [s for s in file_fan]
# polar_fan = [s[1] for s in file_fan]

aspect_keywords = ["appearance", "aroma", "taste", "palate"]

result_yifan = classification_report(aspect_yifan[:1000], predict_aspect_list[:1000],
                                     labels=["appearance", "taste", "palate", "aroma"], digits=2)
print result_yifan

# print '\n'
#
# result_fan = classification_report(aspect_fan[:500], predict_aspect_list[:500],
#                                    labels=["value", "room", "location", "cleanliness", "service"], digits=2)
# print result_fan
