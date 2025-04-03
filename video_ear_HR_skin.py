# This code is based on code from the following GitHub repository:
# Repository: Remote Photoplethysmography to Monitor Human Cardiac Activities 
# Using Commercial Webcams
# Author: Kaushik Goud Chandapet
# URL: https://github.com/kaushik4444/Remote-photoplethysmography-
# to-monitor-Human-cardiac-activities-using-Commercial-Webcams


import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import heartpy as hp
from scipy import signal
import scipy.signal as sig
from xlsxwriter import Workbook

# Butterworth forward-backward band-pass filter
def bandpass(signal, fs, order, fc_low, fc_hig, debug=False):
    """Butterworth forward-backward band-pass filter.

    :param signal: list of ints or floats; The vector containing the signal samples.
    :param fs: float; The sampling frequency in Hz.
    :param order: int; The order of the filter.
    :param fc_low: int or float; The lower cutoff frequency of the filter.
    :param fc_hig: int or float; The upper cutoff frequency of the filter.
    :param debug: bool, default=False; Flag to enable the debug mode that prints additional information.

    :return: list of floats; The filtered signal.
    """
    nyq = 0.5 * fs  # Calculate the Nyquist frequency.
    cut_low = fc_low / nyq  # Calculate the lower cutoff frequency (-3 dB).
    cut_hig = fc_hig / nyq  # Calculate the upper cutoff frequency (-3 dB).
    bp_b, bp_a = sig.butter(order, (cut_low, cut_hig), btype="bandpass")  # Design and apply the band-pass filter.
    bp_data = list(sig.filtfilt(bp_b, bp_a, signal))  # Apply forward-backward filter with linear phase.
    return bp_data


# Fast Fourier Transform
def fft(data, fs, scale="mag"):
    # Apply Hanning window function to the data.
    data_win = data * np.hanning(len(data))
    if scale == "mag":  # Select magnitude scale.
        mag = 2.0 * np.abs(np.fft.rfft(tuple(data_win)) / len(data_win))  # Single-sided DFT -> FFT
    elif scale == "pwr":  # Select power scale.
        mag = np.abs(np.fft.rfft(tuple(data_win))) ** 2  # Spectral power
    bin = np.fft.rfftfreq(len(data_win), d=1.0 / fs)  # Calculate bins, single-sided
    return bin, mag


plt.ion()  # Set interactive mode on
fig = plt.figure(1)
plt.xlabel("Time(ms)")
plt.ylabel("Pixels")
plt.title("Raw RGB signals")

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Excel parameters
time_stamp = []
blue = []
red = []
green = []

# plotting parameters
b_plot = []
g_plot = []
r_plot = []
t_plot = []

# Get source_mp4 Video
source_mp4 = 'data/video/13_math_8.27.mp4'

# Using Video-capture to get the fps value.
capture = cv2.VideoCapture(source_mp4)
fps = capture.get(cv2.CAP_PROP_FPS)
capture.release()

# Using Video-capture to run video file
cap = cv2.VideoCapture(source_mp4)

frame_count = 0  # frames count
time_count = 0  # time in milliseconds
update = 0  # plot update
plot = False  # True to show POS plots
is_update = False

# Skin detection function
def skin_detection(frame):
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create a mask using the skin color range
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Perform morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Extract skin region
    skin = cv2.bitwise_and(frame, frame, mask=mask)
    return skin, mask

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while True:
        success, image = cap.read()
        if image is None:
            break
        height, width, _ = image.shape
        # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        processed_img = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # convert the RGB image to BGR.

        # Perform skin detection
        skin, mask = skin_detection(image)

        # Convert to RGB for plotting
        skin_rgb = cv2.cvtColor(skin, cv2.COLOR_BGR2RGB)

        # Compute mean RGB values
        b_mean, g_mean, r_mean = cv2.split(skin_rgb)
        b_mean = np.mean(b_mean[b_mean > 0])
        g_mean = np.mean(g_mean[g_mean > 0])
        r_mean = np.mean(r_mean[r_mean > 0])

        # Append to plotting lists
        b_plot.append(b_mean)
        g_plot.append(g_mean)
        r_plot.append(r_mean)
        frame_count += 1
        t_plot.append(round(time_count))
        time_count += (1000 / fps)

        # Display results
        cv2.imshow('Original Frame', image)
        cv2.imshow('Skin Region', skin)

        # Plot the graph 4 times a sec (15 new records each time)
        if frame_count % 15 == 0:
            is_update = True  # New frame has come

            # plot the RGB signals
            plt.plot(t_plot, b_plot, 'b', label='Blue')
            plt.plot(t_plot, g_plot, 'g', label='Green')
            plt.plot(t_plot, r_plot, 'r', label='Red')
            plt.pause(0.01)
            update += 1

        elif update > 2:
            # After 3 plots push the reading to Excel parameters and clear plotting parameters
            if is_update:
                if update == 3:
                    blue.extend(b_plot)
                    green.extend(g_plot)
                    red.extend(r_plot)
                    time_stamp.extend(t_plot)
                else:
                    blue.extend(b_plot[(len(b_plot) - 15):len(b_plot)])
                    green.extend(g_plot[(len(g_plot) - 15):len(g_plot)])
                    red.extend(r_plot[(len(r_plot) - 15):len(r_plot)])
                    time_stamp.extend(t_plot[(len(t_plot) - 15):len(t_plot)])

                del b_plot[0:15]
                del g_plot[0:15]
                del r_plot[0:15]
                del t_plot[0:15]

                is_update = False  # we added the new frame to our list structure

        # Break using esc key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
    capture.release()

    # Hold plot and save raw RGB signals
    plt.ioff()
    fig.savefig('result/rPPG_RGB.png', dpi=100)

    # stack r, g, b channels into a single 2-D array
    mean_rgb = np.vstack((red, green, blue)).T

    # Calculating window length l and initiate bvp as 0's
    l = int(fps * 1.6)
    H = np.zeros(mean_rgb.shape[0])

    # POS Algorithm to extract bvp from raw signal
    for t in range(0, (mean_rgb.shape[0] - l)):
        # Step 1: Spatial averaging
        C = mean_rgb[t:t + l - 1, :].T
        # C = mean_rgb.T
        # print("t={0},t+l={1}".format(t, t + l))
        if t == 3:
            plot = False

        if plot:
            f = np.arange(0, C.shape[1])
            plt.plot(f, C[0, :], 'r', f, C[1, :], 'g', f, C[2, :], 'b')
            plt.title("Mean RGB - Sliding Window")
            plt.show()

        # Step 2 : Temporal normalization
        mean_color = np.mean(C, axis=1)
        diag_mean_color = np.diag(mean_color)
        diag_mean_color_inv = np.linalg.inv(diag_mean_color)
        Cn = np.matmul(diag_mean_color_inv, C)
        # Cn = diag_mean_color_inv@C
        # print("Temporal normalization", Cn)

        if plot:
            f = np.arange(0, Cn.shape[1])
            # plt.ylim(0,100000)
            plt.plot(f, Cn[0, :], 'r', f, Cn[1, :], 'g', f, Cn[2, :], 'b')
            plt.title("Temporal normalization - Sliding Window")
            plt.show()

        # Step 3: projection_matrix
        projection_matrix = np.array([[0, 1, -1], [-2, 1, 1]])
        S = np.matmul(projection_matrix, Cn)
        # S = projection_matrix@Cn
        # print("S matrix", S)
        if plot:
            f = np.arange(0, S.shape[1])
            # plt.ylim(0,100000)
            plt.plot(f, S[0, :], 'c', f, S[1, :], 'm')
            plt.title("Projection matrix")
            plt.show()

        # Step 4: 2D signal to 1D signal
        std = np.array([1, np.std(S[0, :]) / np.std(S[1, :])])
        # print("std", std)
        P = np.matmul(std, S)
        # P = std@S
        # print("P", P)
        if plot:
            f = np.arange(0, len(P))
            plt.plot(f, P, 'k')
            plt.title("Alpha tuning")
            plt.show()

        # Step 5: Overlap-Adding
        H[t:t + l - 1] = H[t:t + l - 1] + (P - np.mean(P)) / np.std(P)

    # print("Pulse", H)
    bvp_signal = H
    # print("Raw signal shape", len(green))
    # print("Extracted Pulse shape", H.shape)

    # 2nd order butterworth bandpass filtering
    filtered_pulse = bandpass(bvp_signal, fps, 2, 0.9, 1.8)  # Heart Rate : 60-100 bpm (1-1.7 Hz), taking 54-108 (0.9 - 1.8)
    fig2 = plt.figure(2)
    plt.plot(time_stamp, bvp_signal, 'g', label='Extracted_pulse')
    plt.plot(time_stamp, filtered_pulse, 'r', label='Filtered_pulse')
    plt.title("Raw and Filtered Signals")
    plt.xlabel('Time [ms]')
    # Save the plot as png file
    fig2.savefig('result/rPPG_pulse.png', dpi=100)
    # plt.show()

    # plot welch's periodogram
    bvp_signal = bvp_signal.flatten()
    f_set, f_psd = signal.welch(bvp_signal, fps, window='hamming', nperseg=1024)  # , scaling='spectrum',nfft=2048)
    fig3 = plt.figure(3)
    plt.semilogy(f_set, f_psd)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.title("Welchplot of extracted pulse")
    fig3.savefig('result/rPPG_extractedWelch.png', dpi=100)
    # plt.show()

    # Filtering the welch's periodogram - Heart Rate : 60-100 bpm (1-1.7 Hz), taking 54-108 (0.9 - 1.8)
    # green_psd = green_psd.flatten()
    first = np.where(f_set > 0.9)[0]  # 0.8 for 300 frames
    last = np.where(f_set < 1.9)[0]
    first_index = first[0]
    last_index = last[-1]
    range_of_interest = range(first_index, last_index + 1, 1)

    # get the frequency with highest psd
    # print("Range of interest", range_of_interest)
    max_idx = np.argmax(f_psd[range_of_interest])
    f_max = f_set[range_of_interest[max_idx]]

    # calculate Heart rate
    hr = f_max * 60.0
    print("Detected Heart rate using POS = {0}".format(hr))

    # Calculate and display FFT of filtered pulse
    X_fft, Y_fft = fft(filtered_pulse, fps, scale="mag")
    fig4 = plt.figure(4)
    plt.plot(X_fft, Y_fft)
    plt.title("FFT of filtered Signal")
    plt.xlabel('frequency [Hz]')
    fig4.savefig('result/rPPG_filteredFFT.png', dpi=100)
    # plt.show()

    # Welch's Periodogram of filtered pulse
    f_set, Pxx_den = signal.welch(filtered_pulse, fps, window='hamming', nperseg=1024)
    fig5 = plt.figure(5)
    plt.semilogy(f_set, Pxx_den)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.title("Welchplot of filtered pulse")
    fig5.savefig('result/rPPG_filteredWelch.png', dpi=100)
    # plt.show()

    # Calculate Heart Rate and Plot using HeartPy Library
    working_data, measures = hp.process(filtered_pulse, fps)
    plot_object = hp.plotter(working_data, measures, show=False, title='Final_Heart Rate Signal Peak Detection')
    plot_object.savefig('result/bpmPlotVideo.png', dpi=100)
    peaks = [0] * len(working_data['hr'])
    for p, q in zip(working_data['peaklist'], working_data['binary_peaklist']):
        if q == 1:
            peaks[p] = 1
    detected_peaks_data = {i: peaks.count(i) for i in peaks}
    print('Detected number of peaks = ', detected_peaks_data[1])
    print('Detected Heart rate using HeartPy = ', measures['bpm'])
    print('Detected Inter beat interval using HeartPy =', measures['ibi'])
    print('Detected Breathing rate using HeartPy =', measures['breathingrate']*60)
    plt.show()

    # Export Heart rate (HeartPy) to Excel file
    book = Workbook('result/Heartrate_signal.xlsx')
    sheet = book.add_worksheet()
    row = 0
    col = 0

    sheet.write(row, col, 'Time')
    sheet.write(row, col + 1, 'Signal')
    sheet.write(row, col + 2, 'Peaks')
    row += 1

    for f, b, g in zip(time_stamp, working_data['hr'], peaks):
        sheet.write(row, col, f)
        sheet.write(row, col + 1, b)
        sheet.write(row, col + 2, g)
        row += 1
    book.close()

    # Export Raw RGB signals to Excel file
    book = Workbook('result/RGB_signal.xlsx')
    sheet = book.add_worksheet()
    row = 0
    col = 0

    sheet.write(row, col, 'Time')
    sheet.write(row, col + 1, 'Blue mean')
    sheet.write(row, col + 2, 'Green mean')
    sheet.write(row, col + 3, 'Red mean')
    row += 1

    for f, b, g, r in zip(time_stamp, blue, green, red):
        sheet.write(row, col, f)
        sheet.write(row, col + 1, b)
        sheet.write(row, col + 2, g)
        sheet.write(row, col + 3, r)
        row += 1
    book.close()
