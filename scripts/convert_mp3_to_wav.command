#!/bin/bash
# Navigate to the folder where the script is located
cd "$(dirname "$0")"

# Go to the "sound" subfolder
cd ../data/sound || { echo "❌ Folder '../data/sound' not found!"; exit 1; }

echo "🎧 Converting .mp3 → .wav and deleting originals on success..."
echo

# Loop through all .mp3 files (handles .mp3 and .MP3)
shopt -s nullglob nocaseglob
for f in *.mp3; do
  # Construct target filename
  out="${f%.*}.wav"
  
  # If WAV already exists, skip and DO NOT delete source
  if [ -f "$out" ]; then
    echo "⏭️  Skipping '$f' — '$out' already exists."
    continue
  fi
  
  echo "🔄 Converting '$f' ..."
  # Convert: mono, 16 kHz, 16-bit PCM; quiet errors shown only
  ffmpeg -loglevel error -y -i "$f" -ac 1 -ar 16000 -sample_fmt s16 "$out"
  status=$?
  
  # Only delete source if conversion succeeded and output is non-empty
  if [ $status -eq 0 ] && [ -s "$out" ]; then
    rm -f -- "$f"
    echo "✅ Converted → '$out' and 🗑️ deleted original '$f'"
  else
    echo "❌ Failed to convert '$f' (left original in place)"
    # Clean up partial/bad output if created but empty
    [ -f "$out" ] && [ ! -s "$out" ] && rm -f -- "$out"
  fi
  
  echo
done
shopt -u nullglob nocaseglob

echo "🏁 Done!"
read -p "Press Enter to close..."