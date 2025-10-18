#!/usr/bin/env bash
set -eo pipefail

BASE="${BASE:-http://localhost:8001/transcribe_phonetic}"
CONF="${CONF_THRESHOLD:-0.05}"
MINMS="${MIN_PHONE_MS:-20}"
MIN_INPUT="${MIN_INPUT_MS:-320}"
REJECT="${REJECT_MS:-0}"
LONG_V="${LONG_VOWEL_MS:-140}"
ROOT="${RESULTS_ROOT:-results}"
AUDIO_ROOT="${AUDIO_ROOT:-audio_samples}"

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
RUNTAG="${STAMP}__conf-${CONF}_ms-${MINMS}"
OUTDIR="${ROOT}/${RUNTAG}"
mkdir -p "${OUTDIR}"
mkdir -p "${OUTDIR}/json"
echo "OUTDIR=${OUTDIR}"

cat > "${OUTDIR}/run_meta.json" <<INNER_EOF
{
  "timestamp": "${STAMP}",
  "base_url": "${BASE}",
  "conf_threshold": ${CONF},
  "min_phone_ms": ${MINMS},
  "min_input_ms": ${MIN_INPUT},
  "reject_ms": ${REJECT},
  "long_vowel_ms": ${LONG_V},
  "python": "$(python -V 2>/dev/null || echo "unknown")",
  "git_commit": "$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")",
  "hostname": "$(hostname 2>/dev/null || echo "unknown")"
}
INNER_EOF

# ===== 修正1: ヘッダー変更（avg_conf削除、phones追加）=====
echo "category,filename,http_code,phones_len,phones,kana_text" > "${OUTDIR}/batch_results.csv"

echo "=== カテゴリ別テスト開始 ==="
echo ""

pass=0
total=0

for category_dir in "${AUDIO_ROOT}"/*/; do
  [ -d "$category_dir" ] || continue
  
  category_name=$(basename "$category_dir")
  
  echo "【カテゴリ: $category_name】"
  
  category_pass=0
  category_total=0
  
  for wav_file in "${category_dir}"*.wav; do
    [ -f "$wav_file" ] || continue
    
    filename=$(basename "$wav_file")
    
    resp=$(curl -s -w '\n%{http_code}' -F "audio=@${wav_file};type=audio/wav" "${BASE}")
    code=${resp##*$'\n'}
    body=${resp%$'\n'*}
    
    # JSON保存
    echo "$body" > "${OUTDIR}/json/${category_name}__${filename%.wav}.json"
    
    # ===== 修正2: 音素列追加、avg_conf削除 =====
    if [ "$code" = "200" ]; then
      phones_len=$(echo "$body" | jq '.phones | length // 0')
      phones=$(echo "$body" | jq -r '[.phones[].p] | join(" ")')
      kana_text=$(echo "$body" | jq -r '.kana_text // ""')
    else
      phones_len=0
      phones="HTTP ${code}"
      kana_text="HTTP ${code}"
    fi
    
    echo "${category_name},${filename},${code},${phones_len},\"${phones}\",${kana_text}" >> "${OUTDIR}/batch_results.csv"
    
    total=$((total + 1))
    category_total=$((category_total + 1))
    
    if [ "$code" = "200" ] && [ "$phones_len" -ge 1 ] && [ -n "$kana_text" ]; then
      pass=$((pass + 1))
      category_pass=$((category_pass + 1))
      status="✅"
    else
      status="❌"
    fi
    
    printf "  %s %-25s → %-15s (HTTP %s)\n" "$status" "$filename" "$kana_text" "$code"
  done
  
  echo "  サマリ: ${category_pass}/${category_total}"
  echo ""
done

echo "=== 全体サマリ ==="
echo "成功: ${pass}/${total}"
echo ""

rm -f latest && ln -s "${OUTDIR}" latest

INDEX="${ROOT}/index.csv"
if [ ! -f "${INDEX}" ]; then
  mkdir -p "${ROOT}"
  echo "runtag,pass,total,conf_threshold,min_phone_ms,reject_ms,min_input_ms,long_vowel_ms,git_commit" > "${INDEX}"
fi

echo "${RUNTAG},${pass},${total},${CONF},${MINMS},${REJECT},${MIN_INPUT},${LONG_V},$(jq -r .git_commit "${OUTDIR}/run_meta.json")" >> "${INDEX}"

echo "Saved -> ${OUTDIR}"
echo "Summary -> PASS=${pass}/${total}"

echo ""
echo "=== カテゴリ別成功率 ==="
# ===== 修正3: AWKの列番号変更（$5→$6）=====
awk -F, '
NR > 1 {
  cat[$1]++
  if ($3 == 200 && $4 >= 1 && $6 != "") success[$1]++
}
END {
  for (c in cat) {
    printf "  %-35s %2d/%2d (%.1f%%)\n", c, success[c]+0, cat[c], (success[c]+0)*100/cat[c]
  }
}' "${OUTDIR}/batch_results.csv"

# JSON統合処理
echo ""
echo "=== JSON統合中 ==="

ALL_JSON="${OUTDIR}/all_results.json"
echo "[" > "$ALL_JSON"

first=true
for json_file in "${OUTDIR}"/json/*.json; do
  [ -f "$json_file" ] || continue
  
  filename=$(basename "$json_file" .json)
  
  if [ "$first" = true ]; then
    first=false
  else
    echo "," >> "$ALL_JSON"
  fi
  
  jq --arg fname "$filename" '. + {filename: $fname}' "$json_file" >> "$ALL_JSON"
done

echo "" >> "$ALL_JSON"
echo "]" >> "$ALL_JSON"

json_count=$(jq '. | length' "$ALL_JSON")

echo "✅ JSON統合完了: ${json_count}件"
echo "   ${OUTDIR}/all_results.json"
