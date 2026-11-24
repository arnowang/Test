#!/bin/bash
set -e
cd "$(dirname "$0")"

git checkout -B cutpet-mvp
git config user.name "arnowang"
git config user.email "arnowang@users.noreply.github.com"

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin https://github.com/arnowang/Test.git
else
  git remote add origin https://github.com/arnowang/Test.git
fi

git add -A
git status --short

if ! git diff --cached --quiet; then
  git commit -m "${1:-Update CutPet MVP (Google Drive LoRA + Local Web Demo)}"
fi

git push -u origin cutpet-mvp
echo "✅ 已成功推送至 https://github.com/arnowang/Test/tree/cutpet-mvp"
echo "🚀 Colab 直达链接: https://colab.research.google.com/github/arnowang/Test/blob/cutpet-mvp/colab/pet_lora_mvp_server.ipynb"
