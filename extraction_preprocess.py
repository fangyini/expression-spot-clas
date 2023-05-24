import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import cv2
import dlib
import mat73
import scipy.io


def pol2cart(rho, phi):  # Convert polar coordinates to cartesian coordinates for computation of optical strain
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def computeStrain(u, v):
    u_x = u - pd.DataFrame(u).shift(-1, axis=1)
    v_y = v - pd.DataFrame(v).shift(-1, axis=0)
    u_y = u - pd.DataFrame(u).shift(-1, axis=0)
    v_x = v - pd.DataFrame(v).shift(-1, axis=1)
    os = np.array(np.sqrt(u_x ** 2 + v_y ** 2 + 1 / 2 * (u_y + v_x) ** 2).ffill(1).ffill(0))
    return os

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def extract_preprocess(final_images, k, final_names, dataset_name, fromSaved=True):
    predictor_model = "Utils/shape_predictor_68_face_landmarks.dat"
    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(predictor_model)
    dataset = []
    fromSaved=False
    for video in range(len(final_images)):
        videoName = final_names[video]
        if not fromSaved:
            phase = readPhase(videoName, dataset_name, k)
            phase = phase.transpose(2,3,0,1)
            #amp = amp.transpose(2, 0, 1)
            #amp_average = amp.mean(axis=0)+1.645*amp.std(axis=0) #57,57 X=μ + 1.645σ
            phase_cos = phase[0]
            phase_sin = phase[1] #len, H, W

            OFF_video = []
            for img_count in range(final_images[video].shape[0] - k):
                if (img_count == 0):
                    img1 = final_images[video][img_count]
                    reference_img = img1
                    detect = face_detector(reference_img, 1)
                    next_img = 0  # Loop through the frames until all the landmark is detected
                    while (len(detect) == 0):
                        next_img += 1
                        reference_img = final_images[video][img_count + next_img]
                        detect = face_detector(reference_img, 1)
                    shape = face_pose_predictor(reference_img, detect[0])
                    shape=shape_to_np(shape)
                    for _ in range(0):
                        shape=(shape/2-0.5).astype(int)
                    # x is 0, y is 1
                    pad1 = 16
                    sizeF = 111
                    pad2 = 12
                    # Left Eye
                    x11 = max(shape[36][0] - pad1, 0)
                    y11 = shape[36][1]
                    x12 = shape[37][0]
                    y12 = max(shape[37][1] - pad1, 0)
                    x13 = shape[38][0]
                    y13 = max(shape[38][1] - pad1, 0)
                    x14 = min(shape[39][0] + pad1, sizeF)
                    y14 = shape[39][1]
                    x15 = shape[40][0]
                    y15 = min(shape[40][1] + pad1, sizeF)
                    x16 = shape[41][0]
                    y16 = min(shape[41][1] + pad1, sizeF)

                    # Right Eye
                    x21 = max(shape[42][0] - pad1, 0)
                    y21 = shape[42][1]
                    x22 = shape[43][0]
                    y22 = max(shape[43][1] - pad1, 0)
                    x23 = shape[44][0]
                    y23 = max(shape[44][1] - pad1, 0)
                    x24 = min(shape[45][0] + pad1, sizeF)
                    y24 = shape[45][1]
                    x25 = shape[46][0]
                    y25 = min(shape[46][1] + pad1, sizeF)
                    x26 = shape[47][0]
                    y26 = min(shape[47][1] + pad1, sizeF)

                    # ROI 1 (Left Eyebrow)
                    x31 = max(shape[17][0] - pad2, 0)
                    y32 = max(shape[19][1] - pad2, 0)
                    x33 = min(shape[21][0] + pad2, sizeF)
                    y34 = min(shape[41][1] + pad2, sizeF)

                    # ROI 2 (Right Eyebrow)
                    x41 = max(shape[22][0] - pad2, 0)
                    y42 = max(shape[24][1] - pad2, 0)
                    x43 = min(shape[26][0] + pad2, sizeF)
                    y44 = min(shape[46][1] + pad2, sizeF)

                    # ROI 3 #Mouth
                    x51 = max(shape[60][0] - pad2, 0)
                    y52 = max(shape[50][1] - pad2, 0)
                    x53 = min(shape[64][0] + pad2, sizeF)
                    y54 = min(shape[57][1] + pad2, sizeF)

                    # Nose landmark
                    x61 = shape[28][0]
                    y61 = shape[28][1]

                # phase difference from [video][img_count] to [video][img_count+k]
                cosdiff = phase_cos[(img_count+1):(img_count+k)].sum(axis=0)
                sindiff = phase_sin[(img_count+1):(img_count+k)].sum(axis=0)

                # Eye masking
                left_eye = [(x11, y11), (x12, y12), (x13, y13), (x14, y14), (x15, y15), (x16, y16)]
                right_eye = [(x21, y21), (x22, y22), (x23, y23), (x24, y24), (x25, y25), (x26, y26)]
                cv2.fillPoly(cosdiff, [np.array(left_eye)], 0)
                cv2.fillPoly(cosdiff, [np.array(right_eye)], 0)
                cv2.fillPoly(sindiff, [np.array(left_eye)], 0)
                cv2.fillPoly(sindiff, [np.array(right_eye)], 0)

                final_image = np.zeros((30, 30, 3))
                final_image[:15, :, 0] = cv2.resize(cosdiff[min(y32, y42): max(y34, y44), x31:x43], (30, 15))
                final_image[:15, :, 1] = cv2.resize(sindiff[min(y32, y42): max(y34, y44), x31:x43], (30, 15))
                final_image[15:30, :, 0] = cv2.resize(cosdiff[y52:y54, x51:x53], (30, 15))
                final_image[15:30, :, 1] = cv2.resize(sindiff[y52:y54, x51:x53], (30, 15))
                cosdiff=final_image[:,:,0]
                sindiff=final_image[:,:,1]

                cosdiff=(cosdiff-cosdiff.mean())/cosdiff.std()
                sindiff = (sindiff - sindiff.mean())/sindiff.std()
                magnitude = (cosdiff ** 2 + sindiff ** 2) ** 0.5
                #osx = computeStrain(cosdiff, sindiff)
                final_image[:,:,2]=magnitude

                '''import seaborn as sns
                import matplotlib.pyplot as plt
                ax = sns.heatmap(osx, vmin=0)
                plt.show()
                quit()'''

                # Features Concatenation into 128x128x3
                '''final = np.zeros((30, 30, 3))
                final[:, :, 0] = cv2.resize(cosdiff, (30,30))
                final[:, :, 1] = cv2.resize(sindiff, (30,30))
                final[:, :, 2] = cv2.resize(magnitude, (30, 30))'''

                # Remove global head movement by minus nose region
                #final[:, :, 0] = abs(final[:, :, 0] - final[y61 - 1:y61 + 2, x61 - 1:x61 + 2, 0].mean())
                #final[:, :, 1] = abs(final[:, :, 1] - final[y61 - 1:y61 + 2, x61 - 1:x61 + 2, 1].mean())
                #final[:, :, 2] = abs(final[:, :, 2] - final[y61 - 1:y61 + 2, x61 - 1:x61 + 2, 2].mean())

                '''import seaborn as sns
                import matplotlib.pyplot as plt
                ax = sns.heatmap(final[:,:,2])
                plt.show()
                quit()'''

                OFF_video.append(final_image) #changed to 128,128,3
            # final image: 42,42,3; final: 128,128,3
        '''if not fromSaved:
            np.save(dataset_name+'/graphData_k'+str(k)+'_phase_openface_level2/processed/'+videoName, np.array(OFF_video))
        else:
            OFF_video=np.load(dataset_name+'/graphData_k'+str(k)+'_phase_openface_level2/processed/'+videoName+'.npy')'''
        dataset.append(OFF_video)
        print('Video', dataset_name, video, 'Done')
    print('All Done')
    return dataset


def readMatlab(videoName):
    path = 'CASME_sq/phases/' + videoName + '.mat'
    try:
        riesz_features = mat73.loadmat(path)
    except:
        riesz_features = scipy.io.loadmat(path)
    phase = riesz_features['output'][0]
    amp=riesz_features['output'][1]
    return phase, amp

def readPhase(videoName, dataset_name, k):
    phase = np.load(dataset_name + '/graphData_k' + str(k) + '_phase_openface_level1/' + videoName + '_node.npy')
    return phase
