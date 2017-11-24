from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import os
import logging

input_dir = os.path.join(os.path.dirname(__file__), '..',
                         'runs',
                         'TripAdvisorDoc_Document_DocumentGRU_LSAAC1_MASK',
                         '171014_1507966137', '')
step = 4500


def calc_aspect_f1(input_dir, step):
    predict_aspect_list = list(open(input_dir + str(step) + "_aspect_related_name.out", "r").readlines())
    predict_aspect_list = [s.strip().lower() for s in predict_aspect_list if (len(s) > 0 and s != "\n")]
    predict_aspect_list = ['none' if s == 'all' else s for s in predict_aspect_list]

    aspect_set = set(predict_aspect_list)
    logging.info("Aspect Set: " + str(aspect_set))

    data_dir = os.path.join(os.path.dirname(__file__), '..',
                            'data',
                            'hotel_balance_LengthFix1_3000per',
                            '')

    file_yifan = list(open(data_dir + "test_aspect_0.yifan.aspect", "r").readlines())
    file_yifan = [s.split() for s in file_yifan if (len(s) > 0 and s != "\n")]
    aspect_yifan = [s[0] for s in file_yifan]
    polar_yifan = [s[1] for s in file_yifan]

    file_fan = list(open(data_dir + "test_aspect_0.fan.aspect", "r").readlines())
    file_fan = [s.split() for s in file_fan if (len(s) > 0 and s != "\n")]
    aspect_fan = [s[0] for s in file_fan]
    polar_fan = [s[1] for s in file_fan]

    aspect_set = set(aspect_fan)
    print("fan set:" + str(aspect_set))

    aspect_keywords = [["none"], ["value"], ["room"], ["location"], ["cleanliness"], ["service"]]
    logging.info('\n')

    result_yifan = classification_report(aspect_yifan[:500], predict_aspect_list[:500],
                                         labels=["value", "room", "location", "cleanliness", "service"], digits=2)
    logging.info(result_yifan)

    logging.info('\n')

    result_fan = classification_report(aspect_fan[:600], predict_aspect_list[:600],
                                       labels=["value", "room", "location", "cleanliness", "service"], digits=2)
    logging.info(result_fan)

    yifan_f1 = f1_score(aspect_yifan[:500], predict_aspect_list[:500], average='macro')
    fan_f1 = f1_score(aspect_fan[:600], predict_aspect_list[:600], average='macro')

    return [yifan_f1, fan_f1]
