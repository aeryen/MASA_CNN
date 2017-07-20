from sklearn.metrics import cohen_kappa_score

input_dir = "../data/beer_100k/"

file_yifan = list(open(input_dir + "test_yifan.aspect", "r").readlines())
# file_yifan = [s.split() for s in file_yifan if (len(s) > 0 and s != "\n")]
# aspect_yifan = [s[0] for s in file_yifan]
# polar_yifan = [s[1] for s in file_yifan]
aspect_yifan = [s.strip() for s in file_yifan]

file_fan = list(open(input_dir + "test_fan.aspect", "r").readlines())
file_fan = [s.split() for s in file_fan if (len(s) > 0 and s != "\n")]
for line_index in range(len(file_fan)):
    if len(file_fan[line_index]) != 2:
        print len(file_fan[line_index])
aspect_fan = [s[0] for s in file_fan]
polar_fan = [s[1] for s in file_fan]

aspect_set = set(aspect_yifan)
# polar_set = set(polar_yifan)

print aspect_set
# print polar_set

aspect_set = set(aspect_fan)
polar_set = set(polar_fan)

print aspect_set
print polar_set

print cohen_kappa_score(aspect_yifan[:1000], aspect_fan[:1000], labels=None)
# print cohen_kappa_score(polar_yifan[:1000], polar_fan[:1000], labels=None)
