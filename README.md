# 프로젝트 개발 개요
코로나 사태가 일어난 지 2년이 지났습니다. 마스크 쓰고 밖을 나가는 것이 당연해지기 시작했습니다. 하지만 여전히 마스크를 쓰기를 거부하고 다니는 사람이 많습니다. 코로나 일일 확진자는 7천명을 돌파했지만, 반대로 2년이란 시간동안 사람들은 이 상황에 익숙해져서 코로나를 두려워하는 마음이 사라졌습니다. 신고를 하면 된다고 하지만, 마스크를 안쓴 사람 바로 앞에서 신고를 하기란 어려운일이 아닐 수 없습니다.
이 프로젝트는 그런 상황을 위해 만들었습니다. 마스크 신고 프로그램은 딥 러닝으로 마스크를 쓴 얼굴을 학습하여, 사람들이 마스크 착용 여부를 인식하여 화면에 나타냅니다. 그와 동시에 즉각 미착용자를 문자로 신고하여 빠른 처리가 가능하게 합니다.

<br/>

## 목적 달성을 위한 프로세스 요약
1. Dataset 준비
    1-1. OpneCV 활용 Face detection
    1-2. Facial landmarks detection
    1-3. 배경이 투명한 마스크 이미지를 얼굴에 씌우기

2. Transfer learning - 전이 학습 활용하여 dataset 학습

3. Face detection - 영상에서 사람의 얼굴 도출

4. Detect mask from face – 얼굴에서 마스크 검출
                            마스크를 발견했다면 초록색 사각형,                          
                            마스크를 발견하지 못했다면 5번으로 진행

5. Reprot him – 마스크를 발견하지 못했다면, 빨간색 사각형으로 표시
                  문자를 활용한 신고

## 구성인원
-1인(본인)


## 프로젝트 결과
![image](https://github.com/lovecandy010/mask_check_system/assets/95009128/1f14745a-9796-4280-9243-122fe4970f62)
![image](https://github.com/lovecandy010/mask_check_system/assets/95009128/9859a277-a462-4531-bfda-dca0be1179dd)
###### 개인정보 보호를 위해 얼굴은 가렸습니다


프로그램을 실행 시 기본적으로 Mask, No Mask 두가지 상태로 나뉩니다.
기존 학습된 얼굴과 비교하여 마스크를 썼을 확률을 구하여 상태를 구분합니다.

Mask 상태 :
초록색 사각형으로 얼굴을 인식, 그 외 문제가 없는 정상 상태입니다.

No Mask 상태 :
붉은 사각형으로 얼굴을 인식합니다.
즉각적인 알람과 함께 마스크 미착용자를 신고합니다.

![image](https://github.com/lovecandy010/mask_check_system/assets/95009128/19d6b031-e21a-435f-b546-769f581dca37)
