import os
import whisper
import warnings
import json
import time
import csv

# 경고 메시지 숨김
warnings.filterwarnings("ignore")

def main():
    # 앞서 테스트하셨던 'large' 모델을 사용합니다.
    model_size = "large" 
    
    import torch
    # GPU(CUDA)가 사용 가능한지 확인한 뒤 device를 설정합니다.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper '{model_size}' model on {device.upper()}... (it might take a minute)")
    
    model = whisper.load_model(model_size, device=device)

    sample_dir = "sample_audio"
    if not os.path.exists(sample_dir):
        print("Sample directory not found.")
        return

    # 오디오 파일 리스트 가져오기 및 정렬
    audio_files = [f for f in os.listdir(sample_dir) if f.endswith(".wav")]
    audio_files.sort()
    
    print(f"\nFound {len(audio_files)} audio files. Starting transcription...\n")
    
    # 추출할 메타데이터 키(SpeakerMetadata 내의 속성들)
    meta_keys = [
        "birth_year", "gender", "nationality", "language", "proficiency",
        "topik_score", "topik_level", "education", "learning_period",
        "korea_residency", "purpose", "learning_method", "recording_level"
    ]
    
    # 추가로 추출할 평가 메타데이터 키
    eval_keys = ["DeliveryEval", "LanguageUseEval", "ContentEval"]
    
    # CSV 저장용 리스트
    results = []
    
    start_time = time.time()  # 시작 시간 기록

    for i, file in enumerate(audio_files, 1):
        file_path = os.path.join(sample_dir, file)
        json_file = file_path.replace(".wav", ".json")
        
        # 1. 기본값 설정
        ground_truth = "정답 파일 없음"
        recorded_time = ""
        speaker_metadata = {k: "" for k in meta_keys}
        eval_metadata = {k: "" for k in eval_keys}
        
        # JSON 파일이 존재하면 열어서 데이터 추출
        if os.path.exists(json_file):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # 1. 정답 텍스트(GT) 및 녹음 길이 추출
                rec_meta = data.get("RecordingMetadata", {})
                ground_truth = rec_meta.get("orthographic", "정답 정보 없음")
                recorded_time = rec_meta.get("RecordedTime", "")
                
                # 2. SpeakerMetadata 추출
                meta = data.get("SpeakerMetadata", {})
                for k in meta_keys:
                    speaker_metadata[k] = meta.get(k, "")
                    
                # 3. EvaluationMetadata 추출
                ev_meta = data.get("EvaluationMetadata", {})
                for k in eval_keys:
                    eval_metadata[k] = ev_meta.get(k, "")

        # Whisper를 이용한 예측 수행 (ko 지정 및 fp16=False, 환각 현상 완화 옵션 포함)
        result = model.transcribe(
            file_path, 
            language="ko", 
            fp16=False,
            condition_on_previous_text=False,  # 반복 환각 현상 방지
            no_speech_threshold=0.6  # 침묵이나 잡음을 음성으로 억지로 인식하지 않도록 무시
        )
        prediction = result['text'].strip()
        
        # 대표적인 환각(Hallucination) 문구 강제 삭제 후처리
        prediction = prediction.replace("자막 제공 및 자막 제공 및 광고를 포함하고 있습니다.", "").strip()
        
        # 정답 데이터(GT)에는 없는데 예측 데이터에만 '감사합니다'가 환각으로 나온 경우 제거
        if "감사합니다" not in ground_truth and "감사합니다" in prediction:
            prediction = prediction.replace("감사합니다", "").strip()
        
        print(f"[{i}/{len(audio_files)}] {file} 처리 중...")
        
        # CSV에 저장될 한 행(row) 딕셔너리 만들기
        row_data = {
            "filename": file,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "RecordedTime": recorded_time
        }
        
        # 앞서 JSON에서 추출한 메타데이터 추가
        row_data.update(speaker_metadata)
        row_data.update(eval_metadata)
        
        # 결과 리스트에 담기
        results.append(row_data)

    end_time = time.time()  # 종료 시간 기록
    elapsed = end_time - start_time
    
    # -- 수집된 모든 결과를 CSV 파일로 일괄 저장하기 --
    output_csv = "whisper_results.csv"
    
    # 엑셀에서 바로 열었을 때 한글 깨짐을 방지하기 위해 인코딩으로 'utf-8-sig' 사용
    with open(output_csv, mode="w", encoding="utf-8-sig", newline="") as csvfile:
        # 헤더는 딕셔너리의 키 값들로 지정합니다.
        fieldnames = ["filename", "ground_truth", "prediction", "RecordedTime"] + meta_keys + eval_keys
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 헤더(첫 줄) 먼저 작성
        writer.writeheader()
        
        # 나머지 데이터(행들) 작성
        writer.writerows(results)
    
    # 완료 메시지 출력
    print("\n" + "=" * 60)
    print(f"완료! 총 {len(audio_files)}개 파일 추론 완료.")
    print(f"결과가 현재 폴더의 '{output_csv}' 파일로 정상적으로 저장되었습니다!")
    print(f"총 소요 시간: {elapsed:.2f}초")
    if len(audio_files) > 0:
        print(f"파일 1개당 평균 소요 시간: {elapsed / len(audio_files):.2f}초")
    print("=" * 60)

if __name__ == "__main__":
    main()
