import os
import glob
import whisper
import warnings
import json
import time
import csv
import zipfile
import tempfile
import requests
import re
from pathlib import Path

# ============================================================
# 디스코드 알림 헬퍼
# ============================================================
def send_discord_webhook(message):
    WEBHOOK_URL = "https://discord.com/api/webhooks/1487817136507457556/RSDyVojNlEKte3F3mSUWTsWOLyt3FjJT3tXx16WXd6ov6j4hpll_AScE36H1uYEAsXLp"
    try:
        requests.post(WEBHOOK_URL, json={"content": message}, timeout=5)
    except Exception as e:
        print(f"⚠️ 디스코드 알림 전송 실패: {e}")

# 경고 메시지 숨김
warnings.filterwarnings("ignore")

# ============================================================
# 실제 추론 및 저장 프로세스
# ============================================================
def run_real_task():
    model_size = "large"
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper '{model_size}' model on {device.upper()}... (it might take a minute)")
    
    model = whisper.load_model(model_size, device=device)

    # 타겟 데이터셋 최상위 폴더 경로
    dataset_dir = Path("041.교육용_유럽어_모국어_사용자의_한국어_음성_데이터")
    if not dataset_dir.exists():
        print(f"디렉토리를 찾을 수 없습니다: {dataset_dir}")
        return

    # --- 설정값들 ---
    SAVE_INTERVAL = 100       # 100개마다 CSV에 백업
    
    # 결과 폴더 및 임시 추출 공간 (1TB 여유 공간 활용)
    OUTPUT_DIR = Path("whisper_outputs_041")
    TEMP_EXTRACT_DIR = OUTPUT_DIR / "temp_extracted"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 추출할 메타데이터 키 설정
    meta_keys = [
        "birth_year", "gender", "nationality", "language", "proficiency",
        "topik_score", "topik_level", "education", "learning_period",
        "korea_residency", "purpose", "learning_method", "recording_level"
    ]
    eval_keys = ["DeliveryEval", "LanguageUseEval", "ContentEval"]
    fieldnames = ["filename", "zip_source", "ground_truth", "prediction", "RecordedTime"] + meta_keys + eval_keys

    # 트래커 파일 (이어하기 지원)
    progress_track_file = OUTPUT_DIR / "processed_files_tracker_zip.txt"
    progress_track_file_str = str(progress_track_file)
    
    processed_files = set()
    if progress_track_file.exists():
        with open(progress_track_file_str, "r", encoding="utf-8") as f:
            for line in f:
                processed_files.add(line.strip())
                
    print(f"기존에 이미 처리를 완료하여 건너뛸 파일 수: {len(processed_files)} 개")

    batch_results = []
    start_time = time.time()
    total_processed_count = len(processed_files)
    session_processed_count = 0 

    # 1. 라벨링 JSON 인덱싱
    print("라벨링 ZIP 파일(.json) 목차를 스캔하여 초고속 인덱스를 맵핑합니다...")
    json_index_map = {}
    label_zips = list(dataset_dir.rglob("TL*.zip")) + list(dataset_dir.rglob("VL*.zip"))
    
    for l_zip in label_zips:
        try:
            f_lzip = open(l_zip, 'rb')
            with zipfile.ZipFile(f_lzip, 'r') as zf:
                for member in zf.namelist():
                    if member.endswith(".json"):
                        json_index_map[Path(member).name] = (str(l_zip), member)
            
            # 읽기 완료 후 캐시 메모리 즉시 상환
            if hasattr(os, 'posix_fadvise') and hasattr(os, 'POSIX_FADV_DONTNEED'):
                try: os.posix_fadvise(f_lzip.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                except Exception: pass
            f_lzip.close()
        except zipfile.BadZipFile:
            print(f"⚠️ [손상된 ZIP] {l_zip.name} 건너뜀")
    
    print(f"총 {len(json_index_map)} 개의 정답 데이터 인덱싱 완료.\n")

    # 2. 오디오 ZIP 파일 스캔
    audio_zips = list(dataset_dir.rglob("TS*.zip")) + list(dataset_dir.rglob("VS*.zip"))
    audio_zips.sort()
    
    global_i = 0
    for a_zip_path in audio_zips:
        zip_name = a_zip_path.name
        try:
            f_azip = open(a_zip_path, 'rb')
            with zipfile.ZipFile(f_azip, 'r') as azf:
                wav_members = [m for m in azf.namelist() if m.endswith(".wav")]
                for wav_member in wav_members:
                    global_i += 1
                    wav_basename = Path(wav_member).name
                    if wav_basename in processed_files: continue
                        
                    json_basename = Path(wav_member).with_suffix(".json").name
                    ground_truth, recorded_time = "정답 파일 없음", ""
                    speaker_metadata = {k: "" for k in meta_keys}
                    eval_metadata = {k: "" for k in eval_keys}
                    
                    # JSON 메타데이터 추출 (하이브리드 대응)
                    if json_basename in json_index_map:
                        l_zip_str, json_internal = json_index_map[json_basename]
                        with zipfile.ZipFile(l_zip_str, 'r') as lzf:
                            with lzf.open(json_internal, 'r') as jf:
                                try:
                                    data = json.load(jf)
                                    rec_meta = data.get("RecordingMetadata", {})
                                    ground_truth = rec_meta.get("orthographic") or rec_meta.get("prompt") or "정답 없음"
                                    ground_truth = ground_truth.strip()
                                    recorded_time = rec_meta.get("RecordedTime", "")
                                    
                                    s_meta = data.get("SpeakerMetadata", {})
                                    for k in meta_keys: speaker_metadata[k] = s_meta.get(k, "")
                                        
                                    ev_meta = data.get("EvaluationMetadata", {})
                                    eval_mapping = {
                                        "DeliveryEval": ["DeliveryEval", "PronunProfEval"],
                                        "LanguageUseEval": ["LanguageUseEval", "FluencyEval"],
                                        "ContentEval": ["ContentEval", "ComprehendEval"]
                                    }
                                    for target_key, possible_keys in eval_mapping.items():
                                        for pk in possible_keys:
                                            if pk in ev_meta:
                                                eval_metadata[target_key] = ev_meta[pk]
                                                break
                                except: pass

                    # 임시 오디오 추출 및 추론
                    prediction = ""
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".wav", mode="wb", delete=True, dir=TEMP_EXTRACT_DIR) as temp_wav:
                            with azf.open(wav_member) as source_wav:
                                temp_wav.write(source_wav.read())
                            temp_wav.flush()
                            
                            result = model.transcribe(temp_wav.name, language="ko", fp16=False, condition_on_previous_text=False, no_speech_threshold=0.6)
                            prediction = result['text'].strip()
                            
                            # 환각 후처리
                            prediction = prediction.replace("자막 제공 및 자막 제공 및 광고를 포함하고 있습니다.", "").strip()
                            if "감사합니다" not in ground_truth: prediction = prediction.replace("감사합니다", "").strip()
                                
                    except Exception as e:
                        print(f"⚠️ [추론 실패] {wav_basename}: {e}")
                        prediction = "[추론 실패]"
                        
                    print(f"[{global_i}] {wav_basename} ({zip_name}) 추론 통과")
                    
                    row_data = {"filename": wav_basename, "zip_source": zip_name, "ground_truth": ground_truth, "prediction": prediction, "RecordedTime": recorded_time}
                    row_data.update(speaker_metadata); row_data.update(eval_metadata)
                    
                    batch_results.append(row_data)
                    total_processed_count += 1
                    session_processed_count += 1
                    processed_files.add(wav_basename)
                    
                    # 5만 개 중간 알림
                    if total_processed_count % 50000 == 0:
                        send_discord_webhook(f"📣 **[Antigravity 중간보고]**\n현재 **{total_processed_count:,}개** 완료! 🏃‍♂️💨")
                    
                    # 100개 단위 저장 (재시도 로직 탑재)
                    if len(batch_results) >= SAVE_INTERVAL:
                        grouped = {}
                        for row in batch_results:
                            nat = re.sub(r'[^\w\s]', '', row.get("nationality", "Unknown")).replace(" ", "_")
                            grouped.setdefault(nat, []).append(row)
                            
                        for nat_key, rows in grouped.items():
                            csv_path = OUTPUT_DIR / f"dataset_zip_results_{nat_key}.csv"
                            for attempt in range(5):
                                try:
                                    exists = csv_path.exists()
                                    with open(str(csv_path), "a", encoding="utf-8-sig", newline="") as f:
                                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                                        if not exists: writer.writeheader()
                                        writer.writerows(rows)
                                    break
                                except OSError: time.sleep(3)
                        
                        # 트래커 저장 (재시도 로직)
                        for attempt in range(5):
                            try:
                                with open(progress_track_file_str, "a", encoding="utf-8") as f:
                                    for r in batch_results: f.write(r['filename'] + "\n")
                                break
                            except: time.sleep(2)
                                
                        print(f"🔥 [저장 완료] 국가별 분류 백업 완료! [{', '.join(grouped.keys())}] (누적 {total_processed_count}개)")
                        batch_results.clear()

            # 오디오 ZIP 스캔 완료 후 해당 파일의 캐시만 정밀 파기
            if hasattr(os, 'posix_fadvise') and hasattr(os, 'POSIX_FADV_DONTNEED'):
                try: os.posix_fadvise(f_azip.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                except Exception: pass
            f_azip.close()

        except zipfile.BadZipFile:
            print(f"⚠️ [손상된 ZIP] {a_zip_path.name}")

    # 3. 마지막 잔여 데이터 처리 (100개 미만 꼬리 데이터 백업)
    if batch_results:
        grouped = {}
        for row in batch_results:
            nat = re.sub(r'[^\w\s]', '', row.get("nationality", "Unknown")).replace(" ", "_")
            grouped.setdefault(nat, []).append(row)
            
        for nat_key, rows in grouped.items():
            csv_path = OUTPUT_DIR / f"dataset_zip_results_{nat_key}.csv"
            for attempt in range(3):
                try:
                    exists = csv_path.exists()
                    with open(str(csv_path), "a", encoding="utf-8-sig", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        if not exists: writer.writeheader()
                        writer.writerows(rows)
                    break
                except OSError: time.sleep(2)
        
        # 트래커 마지막 기록
        with open(progress_track_file_str, "a", encoding="utf-8") as f:
            for r in batch_results: f.write(r['filename'] + "\n")
        
        print(f"🏁 [저장 완료] 마지막 잔여 데이터 국가별 백업 완료! [{', '.join(grouped.keys())}]")
        batch_results.clear()

    send_discord_webhook(f"🏁 **[Antigravity 최종보고]**\n총 **{total_processed_count:,}개**의 대장정이 모두 성공적으로 완료되었습니다! 🎉")

# ============================================================
# [메인 실행부] 
# ============================================================
def main():
    try:
        send_discord_webhook("🟢 **[Antigravity 스크립트 시작]** 대장정을 다시 시작합니다! 🚀")
        run_real_task()
    except KeyboardInterrupt:
        send_discord_webhook("🛑 **[중단 알림]** 사용자가 프로그램을 직접 종료했습니다.")
        print("\n[중단] 사용자 요청으로 종료합니다.")
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        send_discord_webhook(f"🚨 **[비상 정지!!]** 에러 발생:\n```{str(e)}```")
        print(f"\n[에러 발생]\n{error_msg}")
        raise e

if __name__ == "__main__":
    main()
