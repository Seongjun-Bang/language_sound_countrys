import os
import glob
import whisper
import warnings
import json
import time
import csv
import zipfile
import tempfile
import re
from pathlib import Path
import traceback

warnings.filterwarnings("ignore")

# 디스코드 웹훅 알림 (필요시 사용)
def send_discord_webhook(message):
    WEBHOOK_URL = "https://discord.com/api/webhooks/1487817136507457556/RSDyVojNlEKte3F3mSUWTsWOLyt3FjJT3tXx16WXd6ov6j4hpll_AScE36H1uYEAsXLp"
    try:
        import requests
        requests.post(WEBHOOK_URL, json={"content": message}, timeout=5)
    except Exception:
        pass

def reprocess_problematic_datasets():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_size = "large"
    print(f"Loading Whisper '{model_size}' model on {device.upper()}...")
    model = whisper.load_model(model_size, device=device)

    # 처리할 데이터셋 매핑
    datasets_to_process = [
        {
            "name": "영어 모국어",
            "prefix": "040",
            "source_dir": "040.교육용_영어_모국어_사용자의_한국어_음성_데이터",
            "output_base": "outputs/영어 모국어"
        },
        {
            "name": "유럽어 모국어",
            "prefix": "041",
            "source_dir": "041.교육용_유럽어_모국어_사용자의_한국어_음성_데이터",
            "output_base": "outputs/유럽어 모국어"
        },
        {
            "name": "중일어 모국어",
            "prefix": "042",
            "source_dir": "042.교육용_중·일어_모국어_사용자의_한국어_음성_데이터",
            "output_base": "outputs/중일어 모국어"
        },
        {
            "name": "아시아어 모국어",
            "prefix": "043",
            "source_dir": "043.교육용_아시아어(중·일어_제외)_사용자의_한국어_음성_데이터",
            "output_base": "outputs/아시아어 모국어"
        }
    ]

    for ds in datasets_to_process:
        # prefix를 기준으로 실제 디렉토리 자동 매칭 (폴더명 불일치 대비)
        prefix = ds.get("prefix")
        source_dir_path = None
        if prefix:
            matches = list(Path(".").glob(f"{prefix}.*"))
            if matches and matches[0].is_dir():
                source_dir_path = matches[0]

        if not source_dir_path:
            source_dir_path = Path(ds['source_dir'])

        if not source_dir_path.exists():
            print(f"⏭️ 데이터셋이 존재하지 않아 {ds['name']} 재처리를 건너뜁니다.")
            continue

        print(f"\n==============================================")
        print(f"🚀 [{ds['name']}] 재처리 시작 (순정 모드)")
        print(f"==============================================")
        
        problematic_dir = Path(ds['output_base']) / "whisper_outputs_problematic"
        reprocessed_dir = Path(ds['output_base']) / "whisper_outputs_reprocessed_fixed"
        reprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        if not problematic_dir.exists():
            continue
            
        csv_files = list(problematic_dir.glob("*.csv"))
        if not csv_files:
            continue
            
        problematic_items = {}
        for csv_file in csv_files:
            with open(csv_file, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "filename" not in row or not row["filename"].strip():
                        continue
                    problematic_items[row["filename"]] = row

        print(f"📌 총 {len(problematic_items)}개의 문제 데이터 발견.")
        if len(problematic_items) == 0:
            continue


        print("🔍 원본 ZIP 파일 스캔 중...")
        audio_zips = list(source_dir_path.rglob("TS*.zip")) + list(source_dir_path.rglob("VS*.zip"))
        
        audio_index = {}
        for a_zip in audio_zips:
            try:
                with zipfile.ZipFile(a_zip, 'r') as azf:
                    for member in azf.namelist():
                        if member.endswith(".wav"):
                            basename = Path(member).name
                            if basename in problematic_items:
                                audio_index[basename] = (a_zip, member)
            except zipfile.BadZipFile:
                pass
                
        found_count = len(audio_index)
        
        temp_extract_dir = reprocessed_dir / "temp"
        temp_extract_dir.mkdir(exist_ok=True)
        
        # 필드명 그대로 유지 (old_prediction 등 추가 없음)
        fieldnames = list(next(iter(problematic_items.values())).keys())

        processed_count = 0
        for filename, row_data in problematic_items.items():
            if filename not in audio_index:
                continue
                
            a_zip, internal_path = audio_index[filename]
            
            try:
                with zipfile.ZipFile(a_zip, 'r') as azf:
                    with tempfile.NamedTemporaryFile(suffix=".wav", mode="wb", delete=True, dir=temp_extract_dir) as temp_wav:
                        with azf.open(internal_path) as source_wav:
                            temp_wav.write(source_wav.read())
                        temp_wav.flush()
                        
                        # 완전히 기존과 동일한 옵션 적용 (프롬프트 제거, temperature 기본)
                        result = model.transcribe(
                            temp_wav.name, 
                            language="ko", 
                            fp16=False, 
                            condition_on_previous_text=False,
                            no_speech_threshold=0.6
                        )
                        prediction = result['text'].strip()
                        
                        prediction = prediction.replace("자막 제공 및 자막 제공 및 광고를 포함하고 있습니다.", "").strip()
                        if "감사합니다" not in row_data.get("ground_truth", ""):
                            prediction = prediction.replace("감사합니다", "").strip()
                            
            except Exception as e:
                print(f"⚠️ [재추론 실패] {filename}: {e}")
                prediction = "[추론 실패]"
            
            # 기존 필드 구조 그대로 덮어쓰기 및 에러 유형 초기화
            row_data["prediction"] = prediction
            row_data["issue_type"] = "" # 새로운 결과이므로 이전 에러 표시 제거
            
            # 국가별 CSV 저장 로직
            nat = re.sub(r'[^\w\s]', '', row_data.get("nationality", "Unknown")).replace(" ", "_")
            csv_path = reprocessed_dir / f"dataset_zip_results_{nat}.csv"
            
            exists = csv_path.exists()
            with open(str(csv_path), "a", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not exists:
                    writer.writeheader()
                writer.writerow(row_data)
            
            processed_count += 1
            print(f"[{processed_count}/{found_count}] {filename} 재추론 완료 (결과: {prediction})")
                
        print(f"✅ [{ds['name']}] 재처리 완료!")

if __name__ == "__main__":
    try:
        send_discord_webhook("🟢 **[문제 데이터 재처리]** 순정 모드로 스크립트 시작!")
        reprocess_problematic_datasets()
        send_discord_webhook("🏁 **[문제 데이터 재처리]** 완료되었습니다!")
    except Exception as e:
        traceback.print_exc()
        send_discord_webhook(f"🚨 **[재처리 에러 발생]**\n```{str(e)}```")
