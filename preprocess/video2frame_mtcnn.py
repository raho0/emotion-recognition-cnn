import os
import time
from cv2 import cv2 
from mtcnn import MTCNN

framepath = './dataset/framesactor3/'
videopath = './dataset/Actor_03/'
framenum = 0
emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

detector = MTCNN()


if not os.path.exists(os.path.join(framepath)):
    os.mkdir(framepath)

#tüm video isimlerini alıyoruz
video_list = os.listdir(videopath)

for i in range(len(video_list)):
    print(f"İşlem: {i+1}/{len(video_list)} ", end="")
    tic = time.perf_counter()
    #videonun üçüncü argümanı duygu durumunu gösteriyor.
    emo = int(video_list[i].split("-")[2]) - 1
    
    #duygular için klasör oluşturuyoruz.
    if not os.path.exists(os.path.join(framepath, emotions[emo])):
        os.mkdir(framepath+emotions[emo])
       
    #videoyu okuyoruz
    videocap = cv2.VideoCapture(os.path.join(*(videopath, video_list[i])))
    success,frame = videocap.read() 
    
    count = 0
    while success:
        videocap.set(cv2.CAP_PROP_POS_MSEC, (count*100))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # yüz algılama
        min_conf = 0.9
        detections = detector.detect_faces(frame)
        for det in detections:
            if det['confidence'] >= min_conf:
                x, y, width, height = det['box']
                sub_face = frame[y:y+height, x:x+width]

            #okunan videonun framelerini ait olduğu sınıfın klasörüne kaydediyoruz.
            facespath = os.path.join(*(framepath, emotions[emo]))+"/frame_"+str(framenum)+".png"
            cv2.imwrite(facespath, sub_face)
            success,frame = videocap.read() 
            framenum += 1
            count += 1
 
    toc = time.perf_counter()
    print(f"{toc - tic:0.2f} saniye sürdü.")