import os
import dlib
import time
from cv2 import cv2 
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb

video_path = './dataset/train_videos/'
framepath = './dataset/frames2dnn/'

framenum = 0
emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor)


if not os.path.exists(os.path.join(framepath)):
    os.mkdir(framepath)

#tüm video isimlerini alıyoruz
video_list = os.listdir(video_path)

for i in range(len(video_list)):
    print(f"İşlem: {i+1}/{len(video_list)} ", end="")
    tic = time.perf_counter()
    #videonun üçüncü argümanı duygu durumunu gösteriyor.
    emo = int(video_list[i].split("-")[2]) - 1
    
    #duygular için klasör oluşturuyoruz.
    if not os.path.exists(os.path.join(framepath, emotions[emo])):
        os.mkdir(framepath+emotions[emo])
       
    #videoyu okuyoruz
    videocap = cv2.VideoCapture(os.path.join(*(video_path, video_list[i])))
    videocap.set(cv2.CAP_PROP_POS_AVI_RATIO,0)
    success,frame = videocap.read()
    
    count = 0
    while success:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # yüz algılama
        rects = detector(gray, 2)
        for rect in rects:
            (x, y, w, h) = rect_to_bb(rect)
            faceAligned = fa.align(frame, gray, rect)

            #okunan videonun framelerini ait olduğu sınıfın klasörüne kaydediyoruz.
            facespath = os.path.join(*(framepath, emotions[emo]))+"/frame_"+str(framenum)+".png"
            cv2.imwrite(facespath, faceAligned)
            success,frame = videocap.read() 
            framenum += 1
            count += 1
 
    toc = time.perf_counter()
    print(f"{toc - tic:0.2f} saniye sürdü.")