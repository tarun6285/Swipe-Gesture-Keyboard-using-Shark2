'''

You can modify the parameters, return values and data structures used in every function if it conflicts with your
coding style or you want to accelerate your code.

You can also import packages you want.

But please do not change the basic structure of this file including the function names. It is not recommended to merge
functions, otherwise it will be hard for TAs to grade your code. However, you can add helper function if necessary.

'''

from flask import Flask, request
from flask import render_template
import time
import json
import math
import matplotlib.pyplot as plt

app = Flask(__name__)

# Centroids of 26 keys
centroids_X = [50, 205, 135, 120, 100, 155, 190, 225, 275, 260, 295, 330, 275, 240, 310, 345, 30, 135, 85, 170, 240, 170, 65, 100, 205, 65]
centroids_Y = [85, 120, 120, 85, 50, 85, 85, 85, 50, 85, 85, 85, 120, 120, 50, 50, 50, 50, 85, 50, 50, 120, 50, 120, 50, 120]

# Pre-process the dictionary and get templates of 10000 words
words, probabilities = [], {}
template_points_X, template_points_Y = [], []
file = open('words_10000.txt')
content = file.read()
file.close()
content = content.split('\n')
for line in content:
    line = line.split('\t')
    words.append(line[0])
    probabilities[line[0]] = float(line[2])
    template_points_X.append([])
    template_points_Y.append([])
    for c in line[0]:
        template_points_X[-1].append(centroids_X[ord(c) - 97])
        template_points_Y[-1].append(centroids_Y[ord(c) - 97])


def generate_sample_points(points_X, points_Y):
    '''Generate 100 sampled points for a gesture.

    In this function, we should convert every gesture or template to a set of 100 points, such that we can compare
    the input gesture and a template computationally.

    :param points_X: A list of X-axis values of a gesture.
    :param points_Y: A list of Y-axis values of a gesture.

    :return:
        sample_points_X: A list of X-axis values of a gesture after sampling, containing 100 elements.
        sample_points_Y: A list of Y-axis values of a gesture after sampling, containing 100 elements.
    '''

    rx = []
    ry = []

    if(len(points_X) == 100):
        return points_X, points_Y
    elif(len(points_X) > 100):

        tlength = len(points_X)
        skip = len(points_X) / (len(points_X) - 100)
        skip = int(skip)

        rx.append(points_X[0])
        ry.append(points_Y[0])

        itr = 1
        while(itr < tlength):
            if(itr % skip != 0):
                rx.append(points_X[itr])
                ry.append(points_Y[itr])
            itr += 1

        while(len(rx) < 100):
            rx.append(points_X[-1])
        while(len(ry) < 100):
            ry.append(points_Y[-1])

        if(len(rx) > 100):
            while(len(rx) > 99):
                rx.pop()
            rx.append(points_X[-1])

        if(len(ry) > 100):
            while(len(ry) > 99):
                ry.pop()
            ry.append(points_Y[-1])

        return rx, ry


    length = 0.0
    count = len(points_X) - 1
    itr = 0
    while(itr < count):
        length = float(length) + math.sqrt((points_X[itr] - points_X[itr + 1])**2 + (points_Y[itr] - points_Y[itr + 1])**2)
        itr += 1

    part = float(length) / 99.0

    if(part == 0.0):
        itr = 0
        while(itr < 100):
            rx.append(points_X[0])
            ry.append(points_Y[0])
            itr += 1
        sample_points_X, sample_points_Y = rx, ry
        return sample_points_X, sample_points_Y

    itr = 0

    py = points_Y[0]
    px = points_X[0]

    while(itr < count):

        wlen = math.sqrt((points_X[itr + 1] - points_X[itr])**2 + (points_Y[itr + 1] - points_Y[itr])**2)

        if (wlen > part):

            wlen = math.sqrt((px - points_X[itr + 1])**2 + (py - points_Y[itr + 1])**2)
            cnt = wlen / part

            cnt = round(cnt, 10)
            cnt = math.floor(cnt)
            itr1 = 0
            while ((itr1 < cnt + 1) and (wlen > 0)):

                rx.append(px)
                ry.append(py)

                px = 1.0 * ((part * points_X[itr + 1]) + ((wlen - part) * px)) / float(wlen)
                py = 1.0 * ((part * points_Y[itr + 1]) + ((wlen - part) * py)) / float(wlen)
                wlen = math.sqrt((px - points_X[itr + 1]) ** 2 + (py - points_Y[itr + 1]) ** 2)

                itr1 += 1

            diffx = abs(px - points_X[itr + 1])
            diffy = abs(py - points_Y[itr + 1])

            if(itr + 1 < count):

                if (points_Y[itr + 2] < points_Y[itr + 1]):
                    py = points_Y[itr + 1] - diffy
                else:
                    py = points_Y[itr + 1] + diffy

                if (points_X[itr + 2] < points_X[itr + 1]):
                    px = points_X[itr + 1] - diffx
                else:
                    px = points_X[itr + 1] + diffx

        itr += 1

    if(len(rx) > 0):

        while(len(rx) < 100):
            rx.append(points_X[-1])
        while(len(ry) < 100):
            ry.append(points_Y[-1])

        if(len(rx) > 100):
            while(len(rx) > 99):
                rx.pop()
            rx.append(points_X[-1])

        if(len(ry) > 100):
            while(len(ry) > 99):
                ry.pop()
            ry.append(points_Y[-1])

    sample_points_X, sample_points_Y = rx, ry
    return sample_points_X, sample_points_Y

# Pre-sample every template
template_sample_points_X, template_sample_points_Y = [], []
for i in range(10000):

    X, Y = generate_sample_points(template_points_X[i], template_points_Y[i])
    template_sample_points_X.append(X)
    template_sample_points_Y.append(Y)

def do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y):
    '''Do pruning on the dictionary of 10000 words.

    In this function, we use the pruning method described in the paper (or any other method you consider it reasonable)
    to narrow down the number of valid words so that the ambiguity can be avoided to some extent.

    :param gesture_points_X: A list of X-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param gesture_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.

    :return:
        valid_words: A list of valid words after pruning.
        valid_probabilities: The corresponding probabilities of valid_words.
        valid_template_sample_points_X: 2D list, the corresponding X-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
        valid_template_sample_points_Y: 2D list, the corresponding Y-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
    '''

    valid_ind = []
    length = len(template_sample_points_Y)
    threshold = 20

    itr = 0
    while(itr < length):

        sdiff = math.sqrt(((gesture_points_X[0] - template_sample_points_X[itr][0])**2) + ((gesture_points_Y[0] - template_sample_points_Y[itr][0])**2))
        ediff = math.sqrt(((gesture_points_X[-1] - template_sample_points_X[itr][-1])**2) + ((gesture_points_Y[-1] - template_sample_points_Y[itr][-1])**2))

        if((sdiff < threshold) and (ediff < threshold)):
            valid_ind.append(itr)
        itr += 1

    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = [], [], []

    length = len(valid_ind)
    itr = 0
    while(itr < length):
        index = valid_ind[itr]
        valid_words.append(words[index])
        valid_template_sample_points_X.append(template_sample_points_X[index])
        valid_template_sample_points_Y.append(template_sample_points_Y[index])
        itr += 1

    return valid_words, valid_template_sample_points_X, valid_template_sample_points_Y


def Normalize(A, B):
    la = len(A)
    suma = 0.0
    mina = float('inf')
    maxa = float('-inf')
    itr = 0
    while(itr < la):
        suma += A[itr]
        if(mina > A[itr]):
            mina = A[itr]
        if(maxa < A[itr]):
            maxa = A[itr]
        itr += 1

    centa = float(suma) / float(la)
    wida = maxa - mina


    lb = len(B)
    sumb = 0.0
    minb = float('inf')
    maxb = float('-inf')
    itr = 0
    while(itr < lb):
        sumb += B[itr]
        if(minb > B[itr]):
            minb = B[itr]
        if(maxb < B[itr]):
            maxb = B[itr]
        itr += 1

    centb = float(sumb) / float(lb)
    widb = maxb - minb

    if(wida < widb):
        wid = widb
    else:
        wid = wida

    itr = 0
    X = []
    Y = []
    while (itr < la):
        if (wid != 0):
            X.append(float(A[itr] - centa) / float(wid))
        else:
            X.append(mina)

        if (wid != 0):
            Y.append(float(B[itr] - centb) / float(wid))
        else:
            Y.append(minb)
        itr += 1

    return X, Y

def get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the shape score for every valid word after pruning.

    In this function, we should compare the sampled input gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a shape score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param valid_template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param valid_template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of shape scores.
    '''

#    gesture_sample_points_Y1 = Normalize(gesture_sample_points_Y)
#    gesture_sample_points_X1 = Normalize(gesture_sample_points_X)
    gesture_sample_points_X1, gesture_sample_points_Y1 = Normalize(gesture_sample_points_X, gesture_sample_points_Y)
    length = len(valid_template_sample_points_X)
    itr = 0
    while(itr < length):
        itr += 1

    shape_scores = []
    length = len(valid_template_sample_points_X)
    itr = 0
    while(itr < length):

        isr = 0
        diff = 0.0
        valid_template_sample_points_X2, valid_template_sample_points_Y2 = Normalize(valid_template_sample_points_X[itr], valid_template_sample_points_Y[itr])
        while(isr < 100):
            val = eud(gesture_sample_points_X1[isr], gesture_sample_points_Y1[isr], valid_template_sample_points_X2[isr], valid_template_sample_points_Y2[isr])
            diff += val
            isr += 1

        diff = diff / 100.0
        shape_scores.append(diff)
        itr += 1

    return shape_scores

def eud(x1, y1, x2, y2):

    return math.sqrt(((y2 - y1)**2) + ((x2 - x1)**2))

def dpq(px, py, qx, qy, r):

    length = len(qx)
    val = eud(px, py, qx[0], qy[0])
    itr = 1
    while(itr < length):
        val = min(val, eud(px, py, qx[itr], qy[itr]))
        itr += 1
    return val

def Dpq(px, py, qx, qy, r):

    length = len(px)
    sum = 0.0
    itr = 0
    while(itr < length):
        val = max((dpq(px[itr], py[itr], qx, qy, r) - r), 0.0)
        if(val > 0):
            return val
        sum += val
        itr += 1
    return sum

def delalpha(gesture_X, gesture_Y, template_X, template_Y):

    delta = []
    isr = 0
    while(isr < 100):
        val = eud(gesture_X[isr], gesture_Y[isr], template_X[isr], template_Y[isr])
        delta.append(val)
        isr += 1

    asum = 0.0
    alpha = []
    val = 50
    while(val >= 1):
        asum += val
        alpha.append(val)
        val -= 1

    val = 1
    while(val <= 50):
        asum += val
        alpha.append(val)
        val += 1

    sum = 0.0
    isr = 0
    while(isr < 100):
        delta[isr] = (delta[isr] * float(alpha[isr])) / float(asum)
        sum += delta[isr]
        isr += 1

    return sum


def get_location_scores(valid_words, gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the location score for every valid word after pruning.

    In this function, we should compare the sampled user gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a location score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of location scores.
    '''
    location_scores = []
    radius = 15

    length = len(valid_template_sample_points_X)
    itr = 0
    while(itr < length):
        val1 = Dpq(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X[itr], valid_template_sample_points_Y[itr], radius)
        if(val1 == 0):
            val2 = Dpq(valid_template_sample_points_X[itr], valid_template_sample_points_Y[itr], gesture_sample_points_X, gesture_sample_points_Y, radius)
        else:
            val2 = 1

        if((val1 == 0) and (val2 == 0)):
            location_scores.append(0)
        else:
            val = delalpha(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X[itr], valid_template_sample_points_Y[itr])
            location_scores.append(val)
        itr += 1

    return location_scores


def get_integration_scores(shape_scores, location_scores):
    integration_scores = []
    # TODO: Set your own shape weight
    shape_coef = 0.5
    # TODO: Set your own location weight
    location_coef = 0.5
    for i in range(len(shape_scores)):
        integration_scores.append(shape_coef * shape_scores[i] + location_coef * location_scores[i])
    return integration_scores


def get_best_word(valid_words, integration_scores):
    '''Get the best word.

    In this function, you should select top-n words with the highest integration scores and then use their corresponding
    probability (stored in variable "probabilities") as weight. The word with the highest weighted integration score is
    exactly the word we want.

    :param valid_words: A list of valid words.
    :param integration_scores: A list of corresponding integration scores of valid_words.
    :return: The most probable word suggested to the user.
    '''

    length = len(integration_scores)
    min_val = float('inf')

    itr = 0
    while(itr < length):
        if(min_val > integration_scores[itr]):
            min_val = integration_scores[itr]
        itr += 1

    best_word = ""
    itr = 0
    while(itr < length):
        if(integration_scores[itr] == min_val):
            if(len(best_word) == 0):
                best_word = valid_words[itr]
            else:
                best_word += " " + valid_words[itr]
        itr += 1

    print "best word ", best_word
    return best_word


@app.route("/")
def init():
    return render_template('index.html')


@app.route('/shark2', methods=['POST'])
def shark2():

    start_time = time.time()
    data = json.loads(request.get_data())


    gesture_points_X = []
    gesture_points_Y = []
    for i in range(len(data)):
        gesture_points_X.append(data[i]['x'])
        gesture_points_Y.append(data[i]['y'])

    gesture_sample_points_X, gesture_sample_points_Y = generate_sample_points(gesture_points_X, gesture_points_Y)

    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = do_pruning(gesture_sample_points_X, gesture_sample_points_Y, template_sample_points_X, template_sample_points_Y)

    shape_scores = get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)

    location_scores = get_location_scores(valid_words, gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)

    integration_scores = get_integration_scores(shape_scores, location_scores)

    best_word = get_best_word(valid_words, integration_scores)

    end_time = time.time()

#    print '{"best_word":"' + best_word + '", "elapsed_time":"' + str(round((end_time - start_time) * 1000, 5)) + 'ms"}'

    return '{"best_word":"' + best_word + '", "elapsed_time":"' + str(round((end_time - start_time) * 1000, 5)) + 'ms"}'


if __name__ == "__main__":
    app.run()
