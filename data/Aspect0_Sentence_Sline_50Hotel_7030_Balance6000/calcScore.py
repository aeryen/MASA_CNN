import sys
import re

aspectID = int(sys.argv[1])
testPrefix = ""
if (len(sys.argv) > 2) & (sys.argv[2] == "test"):
    testPrefix = "test_"

# num_ReviewPerAspect = {}
reviewOfAspect = {}
for ratingIndex in range(0, 5):
    countFilePath = testPrefix + "aspect_" + str(aspectID) + "_rate_" + str(ratingIndex + 1) + ".count"
    print(("loading count file: " + countFilePath))
    lineCountFile = open(countFilePath, "r")
    countLines = lineCountFile.readlines()
    num_Aspect_Review = len(countLines)
    # num_ReviewPerAspect[ratingIndex] = num_Review

    print(("\tRate " + str(ratingIndex + 1) + " Review: " + str(num_Aspect_Review)))

    sentenceCountList = []
    sumLine = 0

    # sum line counts for rating class
    # also keep a sc list so that we can go form actual review struct
    for i in range(0, num_Aspect_Review):
        sumLine = sumLine + int(countLines[i])
        sentenceCountList.append(int(countLines[i]))
    print(("\tTotal lines expected based on count file: " + str(sumLine)))

    sentenceDataFilePath = testPrefix + "aspect_" + str(aspectID) + "_rate_" + str(ratingIndex + 1) + ".txt"
    print(("\tloading sentence file: " + sentenceDataFilePath))
    sentenceFile = open(sentenceDataFilePath, "r")
    sentenceLines = sentenceFile.readlines()
    num_Sentence = len(sentenceLines)

    print(("\tTotal actual sentence: " + str(num_Sentence)))

    if (num_Sentence == sumLine):
        print("\tLine count matched")
    else:
        print("\tLine count mismatch")
        exit(-1)

    reviewSentenceIndex = 0
    ReviewList = []
    for reviewIndex in range(0, num_Aspect_Review):
        sentenceOfReview = []
        for sentenceIndex in range(0, sentenceCountList[reviewIndex]):
            sentenceOfReview.append(sentenceLines[reviewSentenceIndex])
            reviewSentenceIndex += 1
        ReviewList.append(sentenceOfReview)

    reviewOfAspect[ratingIndex] = ReviewList

# load prob file and compare

if (sys.argv[2] == "test"):
    outFile = open("test_label.out", "r")
else:
    outFile = open("train_label.out", "r")

predLines = outFile.readlines()

formatOutputFile = open(testPrefix + "compile.out", "w")

mse = 0.0
# totalReview = 0
# for k, v in reviewOfAspect:
totalReview = sum([len(v) for k, v in list(reviewOfAspect.items())])
globalSentenceIndex = 0

for ratingIndex in range(0, 5):
    currentAspectReviews = reviewOfAspect[ratingIndex]
    num_Aspect_Review = len(currentAspectReviews)
    for reviewIndex in range(0, num_Aspect_Review):
        sentenceCount = len(currentAspectReviews[reviewIndex])
        reviewRatingSum = 0.0
        for i in range(0, sentenceCount):
            # we have to do the same length filter as in cnn
            if len(currentAspectReviews[reviewIndex][i].split()) <= 150:
                line = predLines[globalSentenceIndex]
                score = int(line)
                globalSentenceIndex += 1
            else:
                score = 3

            reviewRatingSum += score
            formatOutputFile.write("\t\t" + currentAspectReviews[reviewIndex][i])

            formatOutputFile.write("\t\t" + str(score) + "\n")

        # for i in range(0, 5):
        #     reviewRatingProb[i] /= sentenceCount
        reviewRatingAvg = reviewRatingSum / sentenceCount
        nearestClass = round(reviewRatingAvg)
        formatOutputFile.write("avgValue: " + str(reviewRatingAvg) + "\tnearest: " + str(nearestClass) + "\n")
        formatOutputFile.write(
            "real: " + str(ratingIndex) + "\t diff: " + str(reviewRatingAvg - ratingIndex) + "\n")

        print('difference: ' + str(reviewRatingAvg - ratingIndex))
        mse += (reviewRatingAvg - ratingIndex) ** 2

        # "c:\t" + str(sentenceCount) + "\t" +
outFile.close
formatOutputFile.close

mse = mse / totalReview
print(mse)
