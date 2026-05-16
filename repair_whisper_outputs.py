import argparse
import csv
import json
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


# 기본 설정
BASE_OUTPUTS_DIR = Path("outputs")
CATEGORIES = ["영어 모국어", "유럽어 모국어", "중일어 모국어", "아시아어 모국어"]
CATEGORY_TO_DATASET_PREFIX = {
    "영어 모국어": "040",
    "유럽어 모국어": "041",
    "중일어 모국어": "042",
    "아시아어 모국어": "043"
}

META_KEYS = [
    "birth_year", "gender", "nationality", "language", "proficiency",
    "topik_score", "topik_level", "education", "learning_period",
    "korea_residency", "purpose", "learning_method", "recording_level",
]
EVAL_KEYS = ["DeliveryEval", "LanguageUseEval", "ContentEval"]
FIELDNAMES = ["tracker_order", "filename", "zip_source", "ground_truth", "prediction", "RecordedTime"] + META_KEYS + EVAL_KEYS + ["issue_type"]
GROUND_TRUTH_PLACEHOLDERS = {"", "정답 없음", "정답 정보 없음", "정답 파일 없음"}
EXPECTED_EVAL_VALUES = {"", "0", "1", "2", "3", "4", "5"}

# 환각 탐지 정규표현식 리스트
HALLUCINATION_PATTERNS = [
    r"자막\s*제공\s*자?", r"광고를\s*포함", r"시청\s*해\s*주셔서\s*감사합니다",
    r"구독\s*과?\s*좋아요", r"채널", r"영상은?\s*여기까지", r"MBC\s*뉴스",
    r"SBS\s*뉴스", r"KBS\s*뉴스", r"다음\s*영상에서\s*만나요", r"도움이\s*되셨다면"
]


def build_json_index_map(dataset_dir):
    json_index_map = {}
    if not dataset_dir or not dataset_dir.exists():
        return json_index_map
        
    label_zips = sorted(dataset_dir.rglob("TL*.zip")) + sorted(dataset_dir.rglob("VL*.zip"))

    for label_zip in label_zips:
        try:
            with zipfile.ZipFile(label_zip, "r") as zf:
                for member in zf.namelist():
                    if member.endswith(".json"):
                        json_index_map[Path(member).name] = (label_zip, member)
        except zipfile.BadZipFile:
            print(f"⚠️ 손상된 라벨 ZIP 건너뜀: {label_zip}")

    return json_index_map


def task_code_from_filename(filename):
    parts = filename.split("-")
    return parts[4] if len(parts) >= 5 else ""


def infer_ground_truth(rec_meta, filename):
    prompt = (rec_meta.get("prompt") or "").strip()
    orthographic = (rec_meta.get("orthographic") or "").strip()
    task_code = task_code_from_filename(filename)

    if task_code.startswith(("ATQ", "LAR", "SPT", "PDT", "RP")):
        return orthographic or prompt or ""
    if task_code.startswith("EIG"):
        return orthographic or prompt or ""
    if task_code.startswith(("RS", "RW")):
        return prompt or orthographic or ""
    return orthographic or prompt or ""


def normalize_json_data(data, filename):
    rec_meta = data.get("RecordingMetadata", {})
    speaker = data.get("SpeakerMetadata", {})
    evaluation = data.get("EvaluationMetadata", {})

    eval_mapping = {
        "DeliveryEval": ["DeliveryEval", "PronunProfEval"],
        "LanguageUseEval": ["LanguageUseEval", "FluencyEval"],
        "ContentEval": ["ContentEval", "ComprehendEval"],
    }

    normalized = {
        "ground_truth": infer_ground_truth(rec_meta, filename),
        "RecordedTime": rec_meta.get("RecordedTime", ""),
    }

    for key in META_KEYS:
        normalized[key] = speaker.get(key, "")

    for target_key, possible_keys in eval_mapping.items():
        value = ""
        for source_key in possible_keys:
            if source_key in evaluation:
                value = evaluation[source_key]
                break
        normalized[target_key] = value

    return normalized


def load_json_payload(filename, json_index_map, zip_cache):
    json_name = filename.replace(".wav", ".json")
    match = json_index_map.get(json_name)
    if not match:
        return None

    zip_path, member = match
    if zip_path not in zip_cache:
        zip_cache[zip_path] = zipfile.ZipFile(zip_path, "r")

    try:
        payload = zip_cache[zip_path].read(member)
        return json.loads(payload.decode("utf-8-sig"))
    except Exception:
        return None


def should_fill_ground_truth(value):
    return (value or "").strip() in GROUND_TRUTH_PLACEHOLDERS


def clean_hallucinations(prediction, ground_truth):
    if not prediction:
        return ""
    
    cleaned = prediction
    for pattern in HALLUCINATION_PATTERNS:
        if re.search(pattern, ground_truth):
            continue
        cleaned = re.sub(pattern, "", cleaned)
    
    if "감사합니다" not in ground_truth:
        cleaned = cleaned.replace("감사합니다", "")
    if "영상" not in ground_truth:
        cleaned = cleaned.replace("영상", "")
    
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if cleaned in {".", ",", "?", "!"}:
        cleaned = ""
        
    return cleaned


def is_problematic(row):
    """행이 비어있음, 언어 혼용, 환각, 또는 추론 실패 조건에 맞는지 검사"""
    prediction = (row.get("prediction") or "").strip()
    ground_truth = (row.get("ground_truth") or "").strip()
    
    # 1. 비어있거나 Null인 경우
    if not prediction:
        return True, "empty_prediction"

    # 2. 추론 실패
    if prediction == "[추론 실패]":
        return True, "inference_failure"
    
    # 3. 언어 혼용 (영문자 포함)
    if re.search("[a-zA-Z]", prediction):
        return True, "mixed_language"

    # 4. 언어 불일치 (한글이 하나도 없는 경우)
    if not re.search("[가-힣]", prediction):
        return True, "language_mismatch"
    
    # 5. 환각 문구
    for pattern in HALLUCINATION_PATTERNS:
        if re.search(pattern, prediction) and not re.search(pattern, ground_truth):
            return True, "hallucination"
            
    return False, ""


def load_tracker_map(tracker_file):
    tracker_map = defaultdict(list)
    if not tracker_file.exists():
        return tracker_map
        
    with open(tracker_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            filename = line.strip()
            if filename:
                tracker_map[filename].append(i)
    return tracker_map


def repair_row(row, json_data, repair_stats, tracker_map, tracker_counters):
    filename = row["filename"]
    if filename in tracker_map:
        idx = tracker_counters[filename]
        if idx < len(tracker_map[filename]):
            row["tracker_order"] = tracker_map[filename][idx]
            tracker_counters[filename] += 1

    if not json_data:
        repair_stats["json_missing"] += 1
        return row

    normalized = normalize_json_data(json_data, row["filename"])

    original_gt = (row.get("ground_truth") or "").strip()
    repaired_gt = normalized["ground_truth"]
    if should_fill_ground_truth(original_gt) and repaired_gt:
        row["ground_truth"] = repaired_gt
        repair_stats["ground_truth_filled"] += 1
    elif should_fill_ground_truth(original_gt) and not repaired_gt:
        repair_stats["ground_truth_still_empty"] += 1

    if str(row.get("RecordedTime", "")).strip() == "" and str(normalized["RecordedTime"]).strip() != "":
        row["RecordedTime"] = normalized["RecordedTime"]
        repair_stats["recorded_time_filled"] += 1

    for key in META_KEYS:
        if str(row.get(key, "")).strip() == "" and str(normalized[key]).strip() != "":
            row[key] = normalized[key]
            repair_stats[f"{key}_filled"] += 1

    for key in EVAL_KEYS:
        if str(row.get(key, "")).strip() == "" and str(normalized[key]).strip() != "":
            row[key] = normalized[key]
            repair_stats[f"{key}_filled"] += 1

    return row


def audit_rows(rows):
    blank_counts = Counter()
    placeholder_counts = Counter()
    unexpected_eval_counts = Counter()
    example_rows = defaultdict(list)

    for row in rows:
        for field in FIELDNAMES:
            value = str(row.get(field, "")).strip()
            if value == "":
                blank_counts[field] += 1
            if field == "ground_truth" and value in GROUND_TRUTH_PLACEHOLDERS:
                placeholder_counts[(field, value)] += 1

        for field in EVAL_KEYS:
            value = str(row.get(field, "")).strip()
            if value not in EXPECTED_EVAL_VALUES:
                unexpected_eval_counts[field] += 1
                if len(example_rows[field]) < 5:
                    example_rows[field].append(f"{row.get('filename', '')}: {value!r}")

    return blank_counts, placeholder_counts, unexpected_eval_counts, example_rows


def print_audit(title, rows):
    blank_counts, placeholder_counts, unexpected_eval_counts, example_rows = audit_rows(rows)

    print("\n" + "=" * 72)
    print(title)
    print(f"총 행 수: {len(rows):,}")
    print("공란 컬럼:")
    for field, count in blank_counts.most_common():
        if count:
            print(f"- {field}: {count:,}")

    print("ground_truth placeholder:")
    if not placeholder_counts:
        print("- 없음")
    else:
        for (_, value), count in placeholder_counts.most_common():
            print(f"- {value!r}: {count:,}")

    print("평가 컬럼 이상값:")
    if not unexpected_eval_counts:
        print("- 없음")
    else:
        for field, count in unexpected_eval_counts.items():
            print(f"- {field}: {count:,}")
            for example in example_rows[field]:
                print(f"  예시: {example}")
    print("=" * 72)


def load_csv_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(csv_path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def process_category(category_name):
    print(f"\n🚀 카테고리 처리 시작: {category_name}")
    
    cat_dir = BASE_OUTPUTS_DIR / category_name
    input_dir = cat_dir / "whisper_outputs_original"
    output_dir = cat_dir / "whisper_outputs_repaired"
    problematic_dir = cat_dir / "whisper_outputs_problematic"
    
    if not input_dir.exists():
        print(f"⚠️ 입력 디렉토리가 없습니다: {input_dir}")
        return

    # 데이터셋 디렉토리 찾기
    prefix = CATEGORY_TO_DATASET_PREFIX.get(category_name)
    dataset_dir = None
    if prefix:
        matches = list(Path(".").glob(f"{prefix}.*"))
        if matches:
            dataset_dir = matches[0]
            print(f"📂 데이터셋 경로 확인됨: {dataset_dir}")
        else:
            print(f"⏭️ 데이터셋(Prefix: {prefix})이 존재하지 않아 {category_name} 작업을 건너뜁니다.")
            return

    csv_paths = sorted(input_dir.glob("dataset_zip_results_*.csv"))
    if not csv_paths:
        print(f"⚠️ 결과 CSV를 찾지 못했습니다: {input_dir}")
        return

    # 출력 디렉토리 정리
    for d in [output_dir, problematic_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    json_index_map = build_json_index_map(dataset_dir) if dataset_dir else {}
    tracker_map = load_tracker_map(input_dir / "processed_files_tracker_zip.txt")
    tracker_counters = defaultdict(int)
    
    zip_cache = {}
    all_original_rows = []
    all_repaired_rows = []
    repair_stats = Counter()
    problematic_stats = Counter()

    try:
        for csv_path in csv_paths:
            rows = load_csv_rows(csv_path)
            repaired_rows = []
            problematic_rows = []

            for row in rows:
                json_data = load_json_payload(row["filename"], json_index_map, zip_cache)
                repaired_row = repair_row(dict(row), json_data, repair_stats, tracker_map, tracker_counters)
                
                # 1. 환각 문구 자동 정제
                original_prediction = repaired_row.get("prediction", "")
                cleaned_prediction = clean_hallucinations(original_prediction, repaired_row.get("ground_truth", ""))
                repaired_row["prediction"] = cleaned_prediction
                
                if original_prediction != cleaned_prediction:
                    repair_stats["sentences_cleaned"] += 1

                # 2. 문제 데이터 탐지 (비어있음, 언어혼용 포함)
                is_prob, prob_type = is_problematic(repaired_row)
                
                if is_prob:
                    repaired_row["issue_type"] = prob_type
                    problematic_rows.append(repaired_row)
                    problematic_stats[prob_type] += 1
                else:
                    repaired_rows.append(repaired_row)

            write_csv_rows(output_dir / csv_path.name, repaired_rows)
            
            if problematic_rows:
                write_csv_rows(problematic_dir / csv_path.name, problematic_rows)
                
            all_original_rows.extend(rows)
            all_repaired_rows.extend(repaired_rows)
            print(f"  - 처리 완료: {csv_path.name} (정상: {len(repaired_rows)}, 문제: {len(problematic_rows)})")
    finally:
        for zf in zip_cache.values():
            zf.close()

    print_audit(f"[{category_name}] 복구 후 결과 진단", all_repaired_rows)
    
    print(f"\n[{category_name}] 문제 데이터 통계:")
    if not problematic_stats:
        print("  - 발견된 문제 없음")
    else:
        for key, count in problematic_stats.most_common():
            print(f"  - {key}: {count:,}")


def main():
    for category in CATEGORIES:
        process_category(category)


if __name__ == "__main__":
    main()
