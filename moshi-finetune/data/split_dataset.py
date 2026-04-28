import os
import json

with open("/home2/s20235100/data_process/AMI_corpus/valid_dataset.json", "r") as f:
    valid_data = json.load(f)

with open("/home2/s20235100/data_process/AMI_corpus/train_dataset.json", "r") as f:
    train_data = json.load(f)

with open("/home2/s20235100/data_process/AMI_corpus/test_dataset.json", "r") as f:
    test_data = json.load(f)

print(len(valid_data))
print(len(train_data))
print(len(test_data))


val_list = list(valid_data.keys())
train_list = list(train_data.keys())
test_list = list(test_data.keys())

print(val_list)
print(train_list)
print(test_list)

# cp data from ./stereo_ami_balanced

os.makedirs("./stereo_ami_balanced_val/data_stereo", exist_ok=True)
os.makedirs("./stereo_ami_balanced_train/data_stereo", exist_ok=True)
os.makedirs("./stereo_ami_balanced_test/data_stereo", exist_ok=True)

for session in val_list:
    os.system(f"cp -r ./stereo_ami_balanced/data_stereo/{session}_stereo.json ./stereo_ami_balanced_val/data_stereo")
    os.system(f"cp -r ./stereo_ami_balanced/data_stereo/{session}_stereo.wav ./stereo_ami_balanced_val/data_stereo")
for session in train_list:
    os.system(f"cp -r ./stereo_ami_balanced/data_stereo/{session}_stereo.json ./stereo_ami_balanced_train/data_stereo")
    os.system(f"cp -r ./stereo_ami_balanced/data_stereo/{session}_stereo.wav ./stereo_ami_balanced_train/data_stereo")
for session in test_list:
    os.system(f"cp -r ./stereo_ami_balanced/data_stereo/{session}_stereo.json ./stereo_ami_balanced_test/data_stereo")
    os.system(f"cp -r ./stereo_ami_balanced/data_stereo/{session}_stereo.wav ./stereo_ami_balanced_test/data_stereo")


# extract element which covers split dataset
with open("./stereo_ami_balanced/data.jsonl", "r") as f:
    jsonl_data = f.readlines()

# 1. 속도와 중복 방지를 위해 "w" 모드로 루프 바깥에서 파일을 엽니다.
# (사전에 stereo_ami_balanced_val 등의 대상 디렉토리가 만들어져 있어야 합니다)
with open("./stereo_ami_balanced_val/data.jsonl", "w") as f_val, \
     open("./stereo_ami_balanced_train/data.jsonl", "w") as f_train, \
     open("./stereo_ami_balanced_test/data.jsonl", "w") as f_test:
     
    for j in jsonl_data:
        try:
            # 2. json 패키지를 활용해 안전하게 데이터를 읽습니다.
            data_dict = json.loads(j)
            path_str = data_dict.get("path", "")
            
            # 3. 세션 이름만 정확하게 추출합니다. 
            # 예: "data_stereo/TS3008d_248_stereo.wav" -> "TS3008d_248"
            filename = path_str.split('/')[-1]  
            session = filename.replace('_stereo.wav', '').replace('_stereo.json', '')
            
            # "stereo_ami/data_stereo/TS3009a_33_stereo.wav" -> "stereo_ami_balanced_train/data_stereo/TS3009a_33_stereo.wav"
            # 4. 분류 및 저장
            if session in val_list:
                f_val.write(j)
            elif session in train_list:
                f_train.write(j)
            elif session in test_list:
                f_test.write(j)
        except Exception as e:
            print(f"라인 파싱 에러 발생: {e}")
