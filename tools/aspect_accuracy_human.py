from sklearn.metrics import classification_report

input_dir = "../output/1467016279/"

predict_aspect_list = list(open(input_dir + "aspect_related_name.out", "r").readlines())
predict_aspect_list = [s.strip().lower() for s in predict_aspect_list if (len(s) > 0 and s != "\n")]
predict_aspect_list = ['none' if s == 'all' else s for s in predict_aspect_list]

input_dir = "../data/hotel_balance_LengthFix1_3000per/"

file_yifan = list(open(input_dir + "test_aspect_0.yifan.aspect", "r").readlines())
file_yifan = [s.split() for s in file_yifan if (len(s) > 0 and s != "\n")]
aspect_yifan = [s[0] for s in file_yifan]
polar_yifan = [s[1] for s in file_yifan]

file_fan = list(open(input_dir + "test_aspect_0.fan.aspect", "r").readlines())
file_fan = [s.split() for s in file_fan if (len(s) > 0 and s != "\n")]
aspect_fan = [s[0] for s in file_fan]
polar_fan = [s[1] for s in file_fan]

aspect_keywords = [["none"], ["value"], ["room"], ["location"], ["cleanliness"], ["service"]]

result_yifan = classification_report(aspect_yifan[:500], predict_aspect_list[:500],
                                     labels=["value", "room", "location", "cleanliness", "service"], digits=2)
print result_yifan

print '\n'

result_fan = classification_report(aspect_fan[:500], predict_aspect_list[:500],
                                   labels=["value", "room", "location", "cleanliness", "service"], digits=2)
print result_fan
