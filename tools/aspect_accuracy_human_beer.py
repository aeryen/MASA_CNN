from sklearn.metrics import classification_report
import os
import logging

input_dir = os.path.join(os.path.dirname(__file__), '..',
                         'runs',
                         'BeerAdvocateDoc_Document_DocumentGRU_LSAAR1',
                         '171023_1508812432', '')
step = 1500


def calc_aspect_f1(input_dir, step):
    predict_aspect_list = list(open(input_dir + str(step) + "_aspect_related_name.out", "r").readlines())
    predict_aspect_list = [s.strip().lower() for s in predict_aspect_list if (len(s) > 0 and s != "\n")]
    predict_aspect_list = ['none' if s == 'all' else s for s in predict_aspect_list]

    aspect_set = set(predict_aspect_list)
    logging.info("Aspect Set: " + str(aspect_set))

    data_dir = os.path.join(os.path.dirname(__file__), '..',
                            'data',
                            'beer_100k',
                            '')

    file_yifan = list(open(data_dir + "test_yifan.aspect", "r").readlines())
    aspect_yifan = [s.strip() for s in file_yifan if (len(s) > 0 and s != "\n")]
    # aspect_yifan = [s.split()[0] for s in aspect_yifan]  # no need to split because i only marked aspect
    # polar_yifan = [s[1] for s in file_yifan]

    aspect_set = set(aspect_yifan)
    print("yifan set:" + str(aspect_set))

    file_fan = list(open(data_dir + "test_fan.aspect", "r").readlines())
    file_fan = [s.split() for s in file_fan if (len(s) > 0 and s != "\n")]
    aspect_fan = [s[0] for s in file_fan]
    # polar_fan = [s[1] for s in file_fan]

    aspect_set = set(aspect_fan)
    print("fan set:" + str(aspect_set))

    aspect_keywords = ["appearance", "aroma", "taste", "palate"]
    logging.info('\n')

    result_yifan = classification_report(aspect_yifan[:1000], predict_aspect_list[:1000],
                                         labels=["appearance", "aroma", "taste", "palate"], digits=2)
    logging.info(result_yifan)

    logging.info('\n')

    result_fan = classification_report(aspect_fan[:1000], predict_aspect_list[:1000],
                                       labels=["appearance", "aroma", "taste", "palate"], digits=2)
    logging.info(result_fan)

    return [result_yifan, result_fan]


calc_aspect_f1(input_dir=input_dir, step=step)

