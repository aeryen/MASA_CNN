import sys
import math

test_file = open(sys.argv[1], "r")
lines = test_file.readlines()
num_lines = len(lines)

pred_file = open(sys.argv[2], "r")
pred_lines = pred_file.readlines()
num_lines_2 = len(lines)

if num_lines != num_lines_2:
    print("Error in running code")
    sys.exit(1)

error = 0.0
correct = 0
incorrect = 0

err_sum = 0.0
three_sum = 0.0
four_sum = 0.0
for i in range(0, num_lines):  # num_lines
    in_l = lines[i]
    pred_l = pred_lines[i]
    in_value = float(in_l.split(" ")[0])
    pred_value = float(pred_l.split(" ")[0])
    #	print(str(pred_value) + " " + str(in_value))
    if (round(pred_value) == in_value):
        correct += 1
    else:
        incorrect += 1
    err_sum += (pred_value - in_value) ** 2
    three_sum += (pred_value - 3.0) ** 2
    four_sum += (pred_value - 4.0) ** 2

err_sum = err_sum / float(num_lines)
three_sum = three_sum / float(num_lines)
four_sum = four_sum / float(num_lines)

print("Testing error")
print("correct " + str(correct))
print("incorrect " + str(incorrect))
print("accuracy " + str(float(correct) / (correct + incorrect)))

print("MSE   " + str(err_sum))
print("MSE 3 " + str(three_sum))
print("MSE 4 " + str(four_sum))
