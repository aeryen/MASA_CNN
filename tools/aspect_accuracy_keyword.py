import re
import numpy as np


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().split()


input_dir = "../output/1467016279/"

predict_aspect_list = list(open(input_dir + "aspect_related.out", "r").readlines())
predict_aspect_list = [int(s) for s in predict_aspect_list if (len(s) > 0 and s != "\n")]

train_content = list(open("../data/hotel_balance_LengthFix1_3000per/" + "test_aspect_0.txt", "r").readlines())
train_content = [s.strip() for s in train_content]
# Split by words
x_text = [clean_str(sent) for sent in train_content]

aspect_keywords = [["overall"], ["value"], ["room"], ["location"], ["clean"], ["service"]]
aspect_keywords_boot = [["overall"], ["value", "price", "money"], ["room", "space"],
                        ["location", "locate"], ["clean", "dirty"], ["service", "manager"]]
aspect_keywords_more = [["overall", "general"],
                        ["value", "deal", "paying", "worth", "pricy", "overprice", "bargain",
                         "cost", "discount", "fee", "price", "cheap", "expensive", "cheaper", "paid"],
                        ["room", "door", "carpet", "house", "furniture", "bathroom", "ready", "comfortable",
                         "decor", "furnish", "open", "hear", "renovation", "light", "rooms", "screen",
                         "mansion", "condition", "bathtub", "bed", "sink", "windows", "kitchen", "quiet",
                         "conditioner", "sleep", "conditioning", "double", "bedroom", "shampoo", "size", "upgrade",
                         "huge", "beds", "floor", "window", "tub", "shower", "air", "soap", "view", "suite",
                         "small", "wall", "large", "room", "pillow", "television", "housekeeping", "queen",
                         "apartment", "space", "king", "modern", "mirror", "twin", "overlook", "spacious",
                         "louver", "toilet", "bath", "toiletry", "balcony", "decorate", "stay", "noise", "chair",
                         "tower", "tv", "book", "suit", "facing", "hairdryer", "night", "square", "courtyard",
                         "channel", "pillows", "upgraded"],
                        ["location", "shop", "parking", "bus", "centre", "location", "street", "shuttle", "beach",
                         "transportation", "close", "city", "center", "airport", "sight-see", "touristy", "subway",
                         "terminal", "shops", "located", "train", "bank", "conference", "place", "pantheon", "central",
                         "station", "tram", "museum", "traffic", "market", "near", "avenue", "shopping", "underground",
                         "wharf", "walk", "wall", "outside", "position", "car", "stop", "supermarket", "walking",
                         "taxi", "tube", "garage", "block", "restaurant", "surround", "minute", "short", "distance",
                         "easy", "transport", "union", "opera", "trip", "boutique", "district", "locate",
                         "restaurants", "downtown", "min", "boulevard", "site", "minutes", "stay", "park", "metro",
                         "attractions", "mins", "blocks", "bloc", "convenient", "great", "route", "square", "plaza",
                         "mall"],
                        ["clean", "maintain", "bathrooms", "heated", "urine", "barrier", "grounds", "tidy",
                         "attentive", "white", "bottled", "kids", "impeccably", "teeth", "professional", "pools",
                         "smoke", "turquoise", "immaculately", "neat", "spotlessly", "smoker", "loungers", "shade",
                         "towels", "linen", "smell", "cigarette", "chairs", "plenty", "musty", "clear", "clean",
                         "pool", "dirty", "slide", "crowded", "nonsmoking", "pressure", "crystal", "exceptionally",
                         "towel", "cleanliness", "bug", "well-maintained", "lazy", "maintained"],
                        ["service", "guide", "highspeed", "manager", "bellman", "e-mail", "luggage", "elevator",
                         "free", "help", "drink", "shelf", "lugged", "smile", "wine", "serve", "rude", "continental",
                         "calls", "wi-fi", "router", "reservation", "pc", "cafe", "eat", "bar", "courteous",
                         "newspaper", "buffets", "computer", "breakfast", "carte", "check-in", "desk", "laundry",
                         "computers", "massage", "wired", "concierge", "waiter", "extremely", "frontdesk",
                         "connectivity", "front", "friendly", "access", "greet", "printer", "route", "club",
                         "broadband", "polite", "high-speed", "hi-speed", "agressive", "buffet", "pcs", "request",
                         "lunch", "gym", "welcome", "connection", "fax", "property", "printing", "emails", "helpful",
                         "dinner", "inform", "stuff", "email", "food", "gem", "management", "checkout", "internet",
                         "modem", "connect", "fitness", "garage", "security", "restaurant", "drinks", "immodium",
                         "network", "staff", "wireless", "ethernet", "facility", "ate", "lounge", "lan", "supply",
                         "printers", "office", "receptionist", "suitcase", "speed", "boarding", "dial-up", "print",
                         "entertainment", "reception", "wifi", "meeting", "service", "cord"]]
aspect_sentence_count = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
correct_count = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

total_sent_correct = 0
total_sent_count = 0

for sentence in x_text:
    no_class = True
    for aspect_index in range(len(aspect_keywords)):
        for word in sentence:
            if word in aspect_keywords_more[aspect_index]:
                if no_class:
                    no_class = False
                    total_sent_count += 1
                aspect_sentence_count[aspect_index] += 1
                sentence_index = x_text.index(sentence)
                if predict_aspect_list[sentence_index] == aspect_index:
                    total_sent_correct += 1
                    correct_count[aspect_index] += 1
                    break

print((str(correct_count)))
print((str(aspect_sentence_count)))
result = np.divide(correct_count, aspect_sentence_count)
print((str(result)))
print()
print("total " + str(total_sent_correct) + " / " + str(total_sent_count))
print("= " + str(total_sent_correct / float(total_sent_count)))
