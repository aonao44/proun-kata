#!/bin/bash

JSON_FILE="latest/all_results.json"

echo "=== 問題箇所の分析 ==="
echo ""

# 期待される音素マッピング
declare -A EXPECTED_PHONES
EXPECTED_PHONES["get_it"]="g ɛ t ɪ t"
EXPECTED_PHONES["apple"]="æ p l"
EXPECTED_PHONES["hello"]="h ɛ l oʊ"
EXPECTED_PHONES["water"]="w ɔ t ər"
EXPECTED_PHONES["coffee"]="k ɔ f i"
EXPECTED_PHONES["happy"]="h æ p i"
EXPECTED_PHONES["banana"]="b ə n æ n ə"
EXPECTED_PHONES["need_in"]="n i d ɪ n"
EXPECTED_PHONES["do_it"]="d u ɪ t"

# 期待されるカタカナ
declare -A EXPECTED_KANA
EXPECTED_KANA["get_it"]="ゲリット"
EXPECTED_KANA["apple"]="アップル"
EXPECTED_KANA["hello"]="ハロー"
EXPECTED_KANA["water"]="ウォーター"
EXPECTED_KANA["coffee"]="コーヒー"
EXPECTED_KANA["happy"]="ハッピー"
EXPECTED_KANA["banana"]="バナナ"
EXPECTED_KANA["need_in"]="ニーディン"
EXPECTED_KANA["do_it"]="ドゥーイット"

echo "【詳細分析】"
echo ""

for key in "${!EXPECTED_PHONES[@]}"; do
  # JSONから該当データを抽出
  data=$(jq -r ".[] | select(.filename | contains(\"$key\"))" "$JSON_FILE")
  
  if [ -z "$data" ]; then
    continue
  fi
  
  actual_phones=$(echo "$data" | jq -r '[.phones[].p] | join(" ")')
  actual_kana=$(echo "$data" | jq -r '.kana_text')
  expected_phones="${EXPECTED_PHONES[$key]}"
  expected_kana="${EXPECTED_KANA[$key]}"
  
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "単語: $key"
  echo ""
  echo "【音素レベル】"
  echo "  期待: $expected_phones"
  echo "  実際: $actual_phones"
  
  if [ "$actual_phones" = "$expected_phones" ]; then
    echo "  判定: ✅ 音素認識は正しい"
    phones_ok=true
  else
    echo "  判定: ❌ 音素認識が間違っている"
    phones_ok=false
  fi
  
  echo ""
  echo "【カタカナレベル】"
  echo "  期待: $expected_kana"
  echo "  実際: $actual_kana"
  
  if [ "$actual_kana" = "$expected_kana" ]; then
    echo "  判定: ✅ カタカナ変換は正しい"
  else
    echo "  判定: ❌ カタカナが期待と異なる"
  fi
  
  echo ""
  echo "【結論】"
  if [ "$phones_ok" = false ]; then
    echo "  🔴 問題: 音素認識（Wav2Vec2）が間違っている"
    echo "     → Edge TTS音声の発音がおかしい可能性"
  elif [ "$actual_kana" != "$expected_kana" ]; then
    echo "  🟡 問題: カタカナ変換ロジックの調整が必要"
  else
    echo "  ✅ 問題なし"
  fi
  
  echo ""
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "=== 総合判定 ==="
echo ""

# 全体の統計
total=$(jq '. | length' "$JSON_FILE")
echo "総サンプル数: $total"
echo ""

# 信頼度の低い音素を検出
echo "【信頼度が低い音素 (conf < 0.3) の割合】"
jq -r '.[] | .filename as $f | .phones[] | select(.conf < 0.3) | "\($f): \(.p) (conf=\(.conf | tostring | .[0:4]))"' "$JSON_FILE" | head -20

echo ""
echo "【信頼度統計】"
jq '[.[] | .phones[].conf] | {avg: (add/length), min: min, max: max}' "$JSON_FILE"

